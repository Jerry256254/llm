"""OS detection, host package install, Docker + NVIDIA runtime preparation."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

# Project root (parent of lib/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKER_DIR = PROJECT_ROOT / "docker"


class Distro(Enum):
    DEBIAN = "debian"  # Debian, Ubuntu, Pop!_OS, etc.
    FEDORA = "fedora"  # Fedora, RHEL, CentOS Stream, Rocky
    UNKNOWN = "unknown"


@dataclass
class GpuInfo:
    name: str
    memory_mib: int
    driver_version: str
    cuda_version: Optional[str]
    index: int


@dataclass
class EnvReport:
    distro: Distro
    distro_pretty: str
    docker_ok: bool
    nvidia_runtime_ok: bool
    gpus: list[GpuInfo]
    cuda_host: Optional[str]


def _run(
    cmd: list[str] | str,
    *,
    check: bool = False,
    capture: bool = True,
    shell: bool = False,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        shell=shell,
        env=env or os.environ.copy(),
    )


def detect_distro() -> tuple[Distro, str]:
    """Detect Debian-family vs Fedora-family Linux."""
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return Distro.UNKNOWN, platform.platform()

    data: dict[str, str] = {}
    for line in os_release.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k] = v.strip().strip('"')

    id_ = data.get("ID", "").lower()
    like = data.get("ID_LIKE", "").lower()
    pretty = data.get("PRETTY_NAME", id_ or "unknown")

    debian_ids = {"debian", "ubuntu", "pop", "linuxmint", "raspbian", "elementary"}
    fedora_ids = {"fedora", "rhel", "centos", "rocky", "almalinux", "ol"}

    if id_ in debian_ids or any(x in like for x in ("debian", "ubuntu")):
        return Distro.DEBIAN, pretty
    if id_ in fedora_ids or any(x in like for x in ("fedora", "rhel", "centos")):
        return Distro.FEDORA, pretty
    return Distro.UNKNOWN, pretty


def _need_sudo() -> bool:
    return os.geteuid() != 0


def _sudo_prefix() -> list[str]:
    return ["sudo"] if _need_sudo() else []


def install_host_packages(distro: Distro) -> None:
    """Install Docker, NVIDIA Container Toolkit, curl, etc. on the host."""
    console.print("[bold cyan]Preparing host packages…[/]")

    if distro == Distro.DEBIAN:
        cmds = [
            "apt-get update -y",
            "apt-get install -y ca-certificates curl gnupg lsb-release "
            "python3-pip python3-venv git pciutils",
        ]
        for c in cmds:
            console.print(f"  $ {c}")
            _run(_sudo_prefix() + ["bash", "-c", c], check=False, capture=False)

        # Docker engine if missing
        if not shutil.which("docker"):
            console.print("[yellow]Installing Docker (get.docker.com)…[/]")
            _run(
                "curl -fsSL https://get.docker.com | " + ("sudo " if _need_sudo() else "") + "sh",
                shell=True,
                capture=False,
            )

        # NVIDIA Container Toolkit
        if not _nvidia_runtime_available():
            console.print("[yellow]Installing NVIDIA Container Toolkit…[/]")
            setup = r"""
set -e
distribution=$(. /etc/os-release; echo ${ID}${VERSION_ID})
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker || true
"""
            _run(_sudo_prefix() + ["bash", "-c", setup], capture=False)

    elif distro == Distro.FEDORA:
        cmds = [
            "dnf install -y curl git python3-pip python3-virtualenv pciutils",
        ]
        for c in cmds:
            console.print(f"  $ {c}")
            _run(_sudo_prefix() + ["bash", "-c", c], check=False, capture=False)

        if not shutil.which("docker"):
            console.print("[yellow]Installing Docker…[/]")
            for c in [
                "dnf -y install dnf-plugins-core",
                "dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo || true",
                "dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin || dnf install -y moby-engine docker",
                "systemctl enable --now docker || true",
            ]:
                _run(_sudo_prefix() + ["bash", "-c", c], check=False, capture=False)

        if not _nvidia_runtime_available():
            console.print("[yellow]Installing NVIDIA Container Toolkit…[/]")
            setup = r"""
set -e
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  tee /etc/yum.repos.d/nvidia-container-toolkit.repo
dnf install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker || true
"""
            _run(_sudo_prefix() + ["bash", "-c", setup], check=False, capture=False)
    else:
        console.print(
            "[red]Unknown distro — install Docker + NVIDIA Container Toolkit manually.[/]"
        )


def _nvidia_runtime_available() -> bool:
    if not shutil.which("docker"):
        return False
    r = _run(["docker", "info"], check=False)
    out = (r.stdout or "") + (r.stderr or "")
    return "nvidia" in out.lower() or Path("/etc/nvidia-container-runtime/config.toml").exists()


def detect_gpus() -> list[GpuInfo]:
    """Parse nvidia-smi for GPU inventory."""
    if not shutil.which("nvidia-smi"):
        return []
    r = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    if r.returncode != 0:
        return []

    gpus: list[GpuInfo] = []
    for line in (r.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_mib=int(float(parts[2])),
                    driver_version=parts[3],
                    cuda_version=None,
                )
            )
        except ValueError:
            continue

    # CUDA version from nvidia-smi header
    r2 = _run(["nvidia-smi"], check=False)
    m = re.search(r"CUDA Version:\s*([\d.]+)", r2.stdout or "")
    cuda = m.group(1) if m else None
    for g in gpus:
        g.cuda_version = cuda
    return gpus


def recommend_cuda_tag(gpus: list[GpuInfo]) -> str:
    """
    Map host driver CUDA capability to a container CUDA image tag.
    We pick a conservative, well-supported Unsloth/torch combo.
    """
    if not gpus or not gpus[0].cuda_version:
        return "12.1.0"
    major_minor = ".".join(gpus[0].cuda_version.split(".")[:2])
    # Supported training images (torch/unsloth ecosystem)
    supported = ["12.4.1", "12.1.0", "11.8.0"]
    try:
        host = tuple(int(x) for x in major_minor.split("."))
        for tag in supported:
            t = tuple(int(x) for x in tag.split(".")[:2])
            if t <= host:
                return tag
    except ValueError:
        pass
    return "12.1.0"


def ensure_docker_group() -> None:
    """Hint if current user cannot talk to Docker daemon."""
    if not shutil.which("docker"):
        return
    r = _run(["docker", "ps"], check=False)
    if r.returncode != 0:
        console.print(
            "[yellow]Docker is installed but not accessible. "
            "Try: sudo usermod -aG docker $USER && newgrp docker[/]"
        )


def build_or_pull_image(
    framework: str = "unsloth",
    cuda_tag: str = "12.1.0",
    force_build: bool = False,
) -> str:
    """
    Build local training image. Image name encodes framework + CUDA.
    """
    image = f"llm-finetune/{framework}:cuda{cuda_tag}"
    dockerfile = DOCKER_DIR / f"Dockerfile.{framework}"
    if not dockerfile.exists():
        raise FileNotFoundError(f"Missing Dockerfile: {dockerfile}")

    exists = _run(["docker", "image", "inspect", image], check=False)
    if exists.returncode == 0 and not force_build:
        console.print(f"[green]Docker image ready:[/] {image}")
        return image

    console.print(f"[bold cyan]Building image {image}…[/] (first run can take a while)")
    cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "--build-arg",
        f"CUDA_VERSION={cuda_tag}",
        "-t",
        image,
        str(PROJECT_ROOT),
    ]
    r = _run(cmd, capture=False)
    if r.returncode != 0:
        raise RuntimeError(f"Docker build failed for {image}")
    return image


def prepare_environment(
    *,
    install_packages: bool = True,
    framework: str = "unsloth",
    force_rebuild_image: bool = False,
) -> EnvReport:
    """Full host + Docker environment bootstrap."""
    distro, pretty = detect_distro()
    console.print(f"[bold]Host OS:[/] {pretty} ({distro.value})")

    if install_packages and distro != Distro.UNKNOWN:
        install_host_packages(distro)

    ensure_docker_group()
    gpus = detect_gpus()
    if not gpus:
        console.print(
            "[red bold]No NVIDIA GPU detected via nvidia-smi.[/]\n"
            "On Google Cloud, use a GPU VM (e.g. n1-standard-8 + T4/L4/A100) "
            "and install the NVIDIA driver."
        )
    else:
        for g in gpus:
            console.print(
                f"  GPU {g.index}: {g.name} — {g.memory_mib} MiB, "
                f"driver {g.driver_version}, CUDA {g.cuda_version or '?'}"
            )

    docker_ok = shutil.which("docker") is not None and _run(["docker", "ps"], check=False).returncode == 0
    nvidia_ok = _nvidia_runtime_available()

    if docker_ok and gpus:
        cuda_tag = recommend_cuda_tag(gpus)
        try:
            build_or_pull_image(framework=framework, cuda_tag=cuda_tag, force_build=force_rebuild_image)
        except Exception as e:
            console.print(f"[red]Image build issue:[/] {e}")

    return EnvReport(
        distro=distro,
        distro_pretty=pretty,
        docker_ok=docker_ok,
        nvidia_runtime_ok=nvidia_ok,
        gpus=gpus,
        cuda_host=gpus[0].cuda_version if gpus else None,
    )


def docker_run_base_args(
    image: str,
    workdir_host: Path,
    *,
    gpus: str = "all",
    extra_env: Optional[dict[str, str]] = None,
    shm_size: str = "16g",
) -> list[str]:
    """Common docker run argv for training / conversion containers."""
    workdir_host = workdir_host.resolve()
    args = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        gpus,
        f"--shm-size={shm_size}",
        "-v",
        f"{workdir_host}:/workspace",
        "-v",
        f"{Path.home() / '.cache' / 'huggingface'}:/root/.cache/huggingface",
        "-w",
        "/workspace",
        "-e",
        "HF_HOME=/root/.cache/huggingface",
        "-e",
        "TRANSFORMERS_CACHE=/root/.cache/huggingface",
    ]
    if extra_env:
        for k, v in extra_env.items():
            args.extend(["-e", f"{k}={v}"])
    args.append(image)
    return args


if __name__ == "__main__":
    report = prepare_environment()
    console.print(report)

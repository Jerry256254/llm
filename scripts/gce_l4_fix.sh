#!/usr/bin/env bash
# Jednorázová oprava GCE G2 + NVIDIA L4 (Debian 12/13 / Ubuntu)
# Spusť:  bash scripts/gce_l4_fix.sh
set -euo pipefail

echo "=== 1) Rozbitý NVIDIA apt list (HTML) ==="
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -y

echo "=== 2) Základní balíčky ==="
sudo apt-get install -y ca-certificates curl gnupg lsb-release \
  python3-pip python3-venv git pciutils build-essential \
  "linux-headers-$(uname -r)" || true

echo "=== 3) Docker ==="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "${USER}" || true

echo "=== 4) NVIDIA ovladač — Google Cloud installer (funguje na Debian 13) ==="
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi už funguje — přeskakuji driver."
else
  # Oficiální GCE GPU installer (ne debian balíček nvidia-driver — na Debian 13 často chybí)
  cd /tmp
  curl -fsSL -o cuda_installer.pyz \
    https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
  sudo python3 cuda_installer.pyz install_driver
  cd - >/dev/null
fi

echo "=== 5) NVIDIA Container Toolkit (stable/deb) ==="
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  if head -1 /etc/apt/sources.list.d/nvidia-container-toolkit.list | grep -qiE '<!doctype|<html'; then
    echo "CHYBA: list soubor je HTML"
    sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
    exit 1
  fi
  sudo apt-get update -y
  sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "=== 6) Kontrola ==="
echo "--- lspci ---"
lspci | grep -iE 'nvidia|3d|vga' || true
echo "--- nvidia-smi ---"
if nvidia-smi; then
  echo "OK: driver běží (L4 by měl být vidět)"
else
  echo
  echo "Driver nainstalován, ale ještě neběží → REBOOT:"
  echo "  sudo reboot"
  echo "Po rebootu:"
  echo "  nvidia-smi"
  echo "  newgrp docker"
  echo "  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
  echo "  cd ~/llm && python3 run.py"
  exit 0
fi

echo "--- docker GPU ---"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi || true

echo
echo "Hotovo. Pokud docker hlásí permission denied:"
echo "  newgrp docker"
echo "Pak: cd ~/llm && python3 run.py"

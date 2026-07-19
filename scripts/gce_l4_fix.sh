#!/usr/bin/env bash
# Jednorázová oprava GCE G2 + NVIDIA L4 (Debian/Ubuntu)
# Spusť na VM:  bash scripts/gce_l4_fix.sh
set -euo pipefail

echo "=== 1) Rozbitý NVIDIA apt list (HTML místo repo) ==="
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -y

echo "=== 2) Základní balíčky ==="
sudo apt-get install -y ca-certificates curl gnupg lsb-release \
  python3-pip python3-venv git pciutils build-essential "linux-headers-$(uname -r)" || true

echo "=== 3) Docker ==="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

echo "=== 4) NVIDIA ovladač (host) pro L4 ==="
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  # Debian: proprietární driver meta-package
  sudo apt-get install -y nvidia-driver || \
  sudo apt-get install -y nvidia-driver-550 || \
  sudo apt-get install -y nvidia-open || true
  if command -v ubuntu-drivers >/dev/null 2>&1; then
    sudo ubuntu-drivers autoinstall || true
  fi
fi

echo "=== 5) NVIDIA Container Toolkit (správná URL stable/deb) ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# kontrola že to není HTML
if head -1 /etc/apt/sources.list.d/nvidia-container-toolkit.list | grep -qiE '<!doctype|<html'; then
  echo "CHYBA: list soubor je HTML — instalace toolkit selhala"
  sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
  exit 1
fi
sudo apt-get update -y
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "=== 6) Kontrola ==="
echo "--- lspci GPU ---"
lspci | grep -i nvidia || lspci | grep -i 3D || true
echo "--- nvidia-smi ---"
if nvidia-smi; then
  echo "OK: driver běží"
else
  echo "Driver ještě neběží → nutný reboot:"
  echo "  sudo reboot"
  echo "Po rebootu: nvidia-smi && docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
  exit 0
fi

echo "--- docker GPU ---"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi || true

echo
echo "Hotovo. Pokud jsi byl právě přidán do skupiny docker:"
echo "  newgrp docker"
echo "  # nebo odhlas/přihlas SSH"
echo "Pak: cd ~/llm && python3 run.py"

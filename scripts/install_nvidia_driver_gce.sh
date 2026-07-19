#!/usr/bin/env bash
# Nainstaluje NVIDIA host driver na GCE (G2/L4, T4, A100, …)
# Debian 12/13 i Ubuntu. Spusť: bash scripts/install_nvidia_driver_gce.sh
set -euo pipefail

echo "=== A) Je na VM vůbec GPU? ==="
if command -v lspci >/dev/null 2>&1; then
  lspci -nn | grep -iE 'nvidia|3d controller|vga' || true
else
  sudo apt-get update -y && sudo apt-get install -y pciutils
  lspci -nn | grep -iE 'nvidia|3d controller|vga' || true
fi

if ! lspci -nn | grep -qi nvidia; then
  echo
  echo "CHYBA: lspci nevidí NVIDIA. GPU není připojené k této VM."
  echo "V Google Cloud Console zkontroluj:"
  echo "  - Machine type = g2-standard-* (L4) nebo n1 + GPU"
  echo "  - VM STOP → Edit → GPU přítomné → Save → START"
  echo "  - Zóna musí GPU podporovat"
  exit 1
fi
echo "OK: NVIDIA zařízení v PCI je vidět."

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi už funguje:"
  nvidia-smi
  exit 0
fi

echo
echo "=== B) Závislosti pro sestavení/instalaci driveru ==="
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates curl python3 build-essential \
  "linux-headers-$(uname -r)" pkg-config || true

echo
echo "=== C) Google Cloud GPU driver installer ==="
cd /tmp
curl -fsSL -o cuda_installer.pyz \
  https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
ls -lh cuda_installer.pyz

# install_driver = jen ovladač (ne celý CUDA toolkit)
set +e
sudo python3 cuda_installer.pyz install_driver
RC=$?
set -e
echo "Installer exit code: $RC"

echo
echo "=== D) Hledám nvidia-smi na disku ==="
sudo find /usr -name 'nvidia-smi' 2>/dev/null | head -20 || true
ls -la /usr/bin/nvidia-smi 2>/dev/null || true
ls -la /usr/local/nvidia/bin/nvidia-smi 2>/dev/null || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== E) Test (před rebootem často selže) ==="
  nvidia-smi || true
fi

echo
echo "============================================================"
echo "POVINNÝ DALŠÍ KROK: reboot, jinak driver nenačte jádro."
echo "  sudo reboot"
echo
echo "Po SSH znovu:"
echo "  nvidia-smi"
echo "  # musí ukázat NVIDIA L4"
echo "============================================================"

#!/bin/bash

set -e  # Exit on any failure

echo "🔧 Updating system and installing core packages..."
sudo apt update && sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev

echo "🔗 Setting python and pip aliases..."
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

echo "✅ Python version:"
python --version
pip --version

echo "📁 Creating workspace directory..."
mkdir -p ~/workspace
cd ~/workspace

echo "📥 Cloning REASON GitHub repository..."
if ! git clone git@github.com:Pajamajoker/REASON.git; then
  echo ""
  echo "❌ ERROR: GitHub SSH clone failed."
  echo ""
  echo "🛠 To fix this, follow these steps to set up SSH keys for GitHub on this VM:"
  echo "------------------------------------------------------------"
  echo "1. Generate SSH key (if not already present):"
  echo "   ssh-keygen -t ed25519 -C \"your-email@example.com\""
  echo ""
  echo "2. Copy the public key:"
  echo "   cat ~/.ssh/id_ed25519.pub"
  echo ""
  echo "3. Go to GitHub → Settings → SSH and GPG keys:"
  echo "   https://github.com/settings/keys"
  echo ""
  echo "4. Click 'New SSH key', paste the key, and save."
  echo ""
  echo "5. Then test with:"
  echo "   ssh -T git@github.com"
  echo ""
  echo "✅ Once done, re-run this script: ./setup-reason-vm.sh"
  echo "------------------------------------------------------------"
  exit 1
fi

echo "📦 Installing Python dependencies..."
cd reason
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ REASON environment is fully set up and ready to use."

#!/bin/bash

echo 'Installing Torch Tuner CLI'

# Detect install prefix: use /opt/homebrew on Apple Silicon Macs, /usr/local otherwise.
INSTALL_PREFIX="/usr/local"
if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
  INSTALL_PREFIX="/opt/homebrew"
fi

# TODO - version argument
# TODO - argument to install from local repo(no git clone)

# Branch is overridable via:
#    * the --branch=<name> CLI arg, e.g.:
#       curl ... | sudo bash -s -- --branch=my-branch
#    * or the BRANCH env var when invoked directly (not via piped `sudo`):
#       sudo BRANCH=my-branch bash ./install-torch-tuner.sh
# Defaults to the repo's default (master).
BRANCH="${BRANCH:-}"

# TODO - install deps. for other OSes
INSTALL_APT_DEPS=0
USER_INSTALL=0
for arg in "$@"; do
  case "$arg" in
     --install-apt-deps) INSTALL_APT_DEPS=1 ;;
     --branch=*) BRANCH="${arg#--branch=}" ;;
     --user) USER_INSTALL=1 ;;
  esac
done

if [[ "$USER_INSTALL" == "1" ]]; then
  INSTALL_PREFIX="$HOME/.local"
fi

if [[ "$INSTALL_APT_DEPS" == "1" ]]; then
  echo 'Installing apt dependencies'
   {
    apt install python3-pip -y && \
      apt install python3-venv -y
   } || {
    echo 'Failed to install Torch Tuner CLI apt dependencies' && \
      exit 1
   }

fi

cd "$INSTALL_PREFIX" || (mkdir -p "$INSTALL_PREFIX/bin" && (cd "$INSTALL_PREFIX" || (echo "failed to create install directory at $INSTALL_PREFIX" && exit 1)))

if [ -d ./torch-tuner ]; then
  echo "Removing old Torch Tuner CLI install"
   {
    rm -rf ./torch-tuner
    if [[ "$USER_INSTALL" == "1" ]]; then
      echo "Removing old Torch Tuner CLI launcher from ~/.local"
      rm -rf "$HOME/.local/bin/torch-tuner"
      rm -rf "$HOME/.local/bin/uninstall-torch-tuner.sh"
    else
      echo "Removing old Torch Tuner CLI launcher"
      rm -rf "$INSTALL_PREFIX/bin/torch-tuner"
      rm -rf "$INSTALL_PREFIX/bin/uninstall-torch-tuner.sh"
    fi
   } || {
    echo 'Failed to remove old Torch Tuner CLI install' && \
    exit 1
   }
fi

if [[ -n "$BRANCH" ]]; then
  echo "Cloning torch-tuner branch: $BRANCH"
   {
    git clone -b "$BRANCH" https://github.com/rjojjr/torch-tuner.git
   } || {
    echo "Failed to clone Torch Tuner CLI branch $BRANCH" && \
      exit 1
   }
else
   {
    git clone https://github.com/rjojjr/torch-tuner.git
   } || {
    echo 'Failed to clone Torch Tuner CLI' && \
      exit 1
   }
fi

# Pick the newest available python>=3.10 on PATH. Required because the locked
# requirements (e.g. absl-py==2.4.0) need >=3.10, and macOS/older distros still
# ship python3 -> 3.9.
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo 'Torch Tuner CLI requires Python >= 3.10, but no compatible python3.1x was found on PATH.'
  if [[ "$(uname)" == "Darwin" ]]; then
    echo 'Install one with Homebrew, e.g.: brew install python@3.12'
  else
    echo 'Install one via your package manager, e.g.: apt install python3.12 python3.12-venv'
  fi
  exit 1
fi
echo "Using $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

{
  cd torch-tuner && \
    "$PYTHON_BIN" -m venv ./.venv && \
    source .venv/bin/activate
} || {
  rm -rf "$INSTALL_PREFIX/torch-tuner" && \
      echo 'Failed to create Torch Tuner CLI venv' && \
      exit 1
}

{
  echo 'Upgrading pip + installing build prerequisites' && \
    pip install --upgrade pip setuptools wheel && \
     {
      if [[ "$(uname)" == "Darwin" ]]; then
        echo 'Installing MPS-compatible PyTorch for macOS (no CUDA build)' && \
          pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 && \
          grep -vE '^(torch==|torchvision==|torchaudio==|triton==|flash-attn==|bitsandbytes==|intel-extension-for-pytorch==|nvidia-)' requirements.in > /tmp/requirements-macos.in && \
          pip install --no-build-isolation -r /tmp/requirements-macos.in && \
          rm /tmp/requirements-macos.in && \
          echo 'Installing HQQ for 4/8-bit quantization on Apple Silicon (bitsandbytes substitute)' && \
          pip install hqq
      else
        echo 'Pre-installing torch (required for flash-attn build)' && \
          grep -E '^torch==' requirements.in | xargs pip install && \
          pip install ninja packaging && \
          echo 'Installing python dependencies' && \
          FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation -I -r requirements.in
      fi
     } && \
    deactivate
} || {
  deactivate 2>/dev/null
  rm -rf "$INSTALL_PREFIX/torch-tuner"
  echo 'Failed to install Torch Tuner CLI python dependencies'
  exit 1
}

{
    cp scripts/torch-tuner "$INSTALL_PREFIX/bin/torch-tuner" && \
      cp scripts/uninstall-torch-tuner.sh "$INSTALL_PREFIX/bin/uninstall-torch-tuner.sh" && \
      chmod -R 755 "$INSTALL_PREFIX/torch-tuner" && \
      chmod +x "$INSTALL_PREFIX/bin/torch-tuner" && \
      chmod +x "$INSTALL_PREFIX/bin/uninstall-torch-tuner.sh"
} || {
  rm -rf "$INSTALL_PREFIX/torch-tuner" && rm "$INSTALL_PREFIX/bin/torch-tuner" && rm "$INSTALL_PREFIX/bin/uninstall-torch-tuner.sh"
  echo 'Failed to install Torch Tuner CLI bash cmd in /bin'
}

echo 'Torch Tuner CLI installed successfully!'
echo "You can now access the Torch Tuner CLI with the 'torch-tuner' command."

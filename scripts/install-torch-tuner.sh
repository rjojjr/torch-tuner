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
#   * the --branch=<name> CLI arg, e.g.:
#       curl ... | sudo bash -s -- --branch=my-branch
#   * or the BRANCH env var when invoked directly (not via piped `sudo`):
#       sudo BRANCH=my-branch bash ./install-torch-tuner.sh
# Defaults to the repo's default (master).
BRANCH="${BRANCH:-}"

# TODO - install deps. for other OSes
INSTALL_APT_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --install-apt-deps) INSTALL_APT_DEPS=1 ;;
    --branch=*) BRANCH="${arg#--branch=}" ;;
  esac
done

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
    if [ -d "$INSTALL_PREFIX/bin/torch-tuner" ]; then
      echo "Removing old Torch Tuner CLI launcher(REQUIRES SUDO)"
      {
        sudo rm "$INSTALL_PREFIX/bin/torch-tuner"
      } || {
        echo 'Failed to remove old Torch Tuner CLI launcher' && \
        exit 1
      }
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

{
  cd torch-tuner && \
    python3 -m venv ./.venv && \
    source .venv/bin/activate
} || {
  rm -rf "$INSTALL_PREFIX/torch-tuner" && \
    echo 'Failed to create Torch Tuner CLI venv' && \
      exit 1
}

{
  echo 'Upgrading pip + installing build prerequisites' && \
    pip install --upgrade pip setuptools wheel && \
    echo 'Pre-installing torch and build helpers (required for flash-attn build)' && \
    grep -E '^torch==' requirements.in | xargs pip install && \
    pip install ninja packaging && \
    echo 'Installing python dependencies' && \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation -I -r requirements.in && \
    deactivate
} || {
  deactivate 2>/dev/null
  rm -rf "$INSTALL_PREFIX/torch-tuner"
  echo 'Failed to install Torch Tuner CLI python dependencies'
  exit 1
}

{
    cp scripts/torch-tuner "$INSTALL_PREFIX/bin/torch-tuner" && \
      chmod -R 755 "$INSTALL_PREFIX/torch-tuner" && \
      chmod +x "$INSTALL_PREFIX/bin/torch-tuner"
} || {
  rm -rf "$INSTALL_PREFIX/torch-tuner" && rm "$INSTALL_PREFIX/bin/torch-tuner"
  echo 'Failed to install Torch Tuner CLI bash cmd in /bin'
}

echo 'Torch Tuner CLI installed successfully!'
echo "You can now access the Torch Tuner CLI with the 'torch-tuner' command."
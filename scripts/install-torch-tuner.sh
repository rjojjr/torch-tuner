#!/bin/bash

echo 'Installing Torch Tuner CLI'

# TODO - version argument
# TODO - argument to install from local repo(no git clone)

# TODO - install deps. for other OSes
if [[ "$1" == "--install-apt-deps" ]]; then
  echo 'Installing apt dependencies'
  {
    apt install python3-pip -y && \
      apt install python3-venv -y
  } || {
    echo 'Failed to install Torch Tuner CLI apt dependencies' && \
      exit 1
  }

fi

cd /usr/local || (mkdir -p /usr/local && (cd /usr/local || (echo 'failed to create install directory at /usr/local' && exit 1)))

if [ -d ./torch-tuner ]; then
  echo "Removing old Torch Tuner CLI install"
  {
    rm -rf ./torch-tuner
  } || {
    echo 'Failed to remove old Torch Tuner CLI install' && \
    exit 1
  }
fi

{
  git clone https://github.com/rjojjr/torch-tuner.git
} || {
  echo 'Failed to clone Torch Tuner CLI' && \
    exit 1
}

{
  cd torch-tuner && \
    python -m venv ./.venv && \
    source .venv/bin/activate
} || {
  rm -rf /usr/local/torch-tuner && \
    echo 'Failed to create Torch Tuner CLI venv' && \
      exit 1
}

{
  echo 'Installing python dependencies' && \
    pip install -r requirements.txt && \
    deactivate
} || {
  deactivate && \
    rm -rf /usr/local/torch-tuner && \
    echo 'Failed to install Torch Tuner CLI python dependencies' && \
    exit 1
}

{
    cp scripts/torch-tuner /bin/torch-tuner && \
      chmod +x /bin/torch-tuner && \
      chmod -R 755 /usr/local/torch-tuner
} || {
  rm -rf /usr/local/torch-tuner && \
  echo 'Failed to install Torch Tuner CLI bash cmd in /bin'
}

echo 'Torch Tuner CLI installed successfully!'
echo "You can now access the Torch Tuner CLI with the 'torch-tuner' command."
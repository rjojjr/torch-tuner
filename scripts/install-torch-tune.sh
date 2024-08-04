#!/bin/bash

echo 'Installing Torch Tuner CLI'

if [[ "$1" == "--install-apt-deps" ]]; then
  echo 'Installing apt dependencies'
  {
    apt install python3-pip -y && \
      apt install python3.10-venv -y
  } || {
    echo 'Failed to install Torch Tuner CLI apt dependencies' && \
      exit 1
  }

fi

cd /usr/local || (mkdir -p /usr/local && (cd /usr/local || (echo 'failed to create install directory at /usr/local' && exit 1)))

if [ -d /var/local/torch-tuner ]; then
  echo 'Removing deprecated Torch Tuner CLI install'
  {
    rm -rf /var/local/torch-tuner
  } || {
    echo 'Failed to remove deprecated Torch Tuner CLI install' && \
      exit 1
  }
fi

export_path="true"
if [ -d ./torch-tuner ]; then
  export_path="false"
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
    chmod +x scripts/torch-tuner
} || {
  deactivate && \
    rm -rf /usr/local/torch-tuner && \
    echo 'Failed to install Torch Tuner CLI python dependencies' && \
    exit 1
}

if [[ "$export_path" == "true" ]]; then
  echo "Adding torch-tuner to current user's path variable"
  echo 'export PATH=$PATH:/usr/local/torch-tuner/scripts' >> ~/.bashrc
  echo 'You will need to run `source ~/.bashrc` or start a new shell session'
fi

echo 'Torch Tuner CLI installed successfully!'
echo "You can now access the Torch Tuner CLI with the 'torch-tuner' command."
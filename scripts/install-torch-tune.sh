#!/bin/bash

echo 'Installing Torch Tuner CLI'

echo 'Installing apt dependencies'
apt install python3-pip -y
apt install python3.10-venv -y

cd /var/local || (mkdir -p /var/local && (cd /var/local || (echo 'failed to create install directory at /var/local' && exit 100)))

export_path="true"
if [ -d ./torch-tuner ]; then
  echo "Removing old install"
  rm -rf ./torch-tuner
  export_path="false"
fi
git clone https://github.com/rjojjr/torch-tuner.git
cd torch-tuner || (echo 'failed to clone repo' && exit 101)
git checkout installer-script
python -m venv ./.venv
source .venv/bin/activate
echo 'Installing python dependencies'
pip install -r requirements.txt
chmod +x scripts/torch-tuner

if [[ "$export_path" == "true" ]]; then
  echo "Adding torch-tuner to current user's path variable"
  echo 'export PATH=$PATH:/var/local/torch-tuner/scripts' >> ~/.bashrc
fi

echo "You can now access the Torch Tuner CLI with the 'torch-tuner' command."

deactivate



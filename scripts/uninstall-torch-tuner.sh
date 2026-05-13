#!/bin/bash

echo 'Uninstalling Torch Tuner CLI'

# Detect install prefix: use /opt/homebrew on Apple Silicon Macs, /usr/local otherwise.
INSTALL_PREFIX="/usr/local"
if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
  INSTALL_PREFIX="/opt/homebrew"
fi

rm -rf "$INSTALL_PREFIX/bin/torch-tuner"
rm -rf "$INSTALL_PREFIX/bin/uninstall-torch-tuner.sh"
rm -rf "$INSTALL_PREFIX/torch-tuner"
echo 'Uninstalled Torch Tuner CLI'
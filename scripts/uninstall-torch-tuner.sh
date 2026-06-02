#!/bin/bash

echo 'Uninstalling Torch Tuner CLI'

# Detect install prefix: use /opt/homebrew on Apple Silicon Macs, /usr/local otherwise.
INSTALL_PREFIX="/usr/local"
if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
  INSTALL_PREFIX="/opt/homebrew"
fi

USER_INSTALL=0
for arg in "$@"; do
  case "$arg" in
      --user) USER_INSTALL=1 ;;
  esac
done

rm -rf "$INSTALL_PREFIX/bin/torch-tuner"
rm -rf "$INSTALL_PREFIX/bin/uninstall-torch-tuner.sh"
rm -rf "$INSTALL_PREFIX/torch-tuner"

if [[ "$USER_INSTALL" == "1" ]]; then
  echo "Also cleaning up ~/.local"
  rm -rf "$HOME/.local/bin/torch-tuner"
  rm -rf "$HOME/.local/bin/uninstall-torch-tuner.sh"
  rm -rf "$HOME/.local/torch-tuner"
fi

echo 'Uninstalled Torch Tuner CLI'

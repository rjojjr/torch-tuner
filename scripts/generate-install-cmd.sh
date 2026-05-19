#!/usr/bin/env bash
# Generate an install command for the current git branch.
# Usage: ./scripts/generate-install-cmd.sh

set -euo pipefail

REPO="rjojjr/torch-tuner"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

CMD="curl -fsSL https://raw.githubusercontent.com/${REPO}/${BRANCH}/scripts/install-torch-tuner.sh | sudo bash -s -- --branch=${BRANCH}"

echo "${CMD}"

#!/usr/bin/env bash

set -e

GIT_ROOT=$(git rev-parse --show-toplevel)

pushd "$GIT_ROOT" > /dev/null

# Download the datasets
[[ ! -f chest-xray-pneumonia.zip ]] && kaggle datasets download -d paultimothymooney/chest-xray-pneumonia || echo "Dataset already downloaded."

[[ -d data ]] && rm -rf data

command -v unzip > /dev/null 2>&1 && unzip -o -d data chest-xray-pneumonia.zip || {
    echo >&2 "Requires unzip, but unzip is not found on PATH. Aborting."
    exit 1
}

popd > /dev/null

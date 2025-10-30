#!/bin/bash

set -euo pipefail

CUR_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(realpath ${CUR_DIR}/../..)
export PYTHONPATH=${ROOT_DIR}:${PYTHONPATH:-}

if [ $# -lt 1 ]; then
  echo "Usage: bash infer.sh <yaml_config> [extra args]"
  exit 1
fi

YAML_CONFIG=$1
shift || true

python3 ${CUR_DIR}/infer.py \
  --yaml_file_path ${YAML_CONFIG} \
  "$@"

#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Bootstrap the driver interpreter from the same YAML used by the launcher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
config=examples/configs/benchmark_eval.yaml
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == --config ]]; then config="${args[i+1]}"; fi
  if [[ "${args[i]}" == --config=* ]]; then config="${args[i]#--config=}"; fi
done
# Host Python has PyYAML; the worker/client run under the existing driver environment.
driver_python=$(python3 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["python"])' "$config")
exec "$driver_python" tools/benchmark_eval/evaluate.py submit "$@"

#!/bin/bash
set -euo pipefail
OWN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 -u "$OWN/forkserver-test.py"

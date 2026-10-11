#!/bin/bash
set -euo pipefail
RUNTIME=/p/vast1/izzet/lfmr/liveflow-campaign-v4.0.6-agentperf-20261010
TREE=$RUNTIME/source
source "$TREE/liveflow_shm/tests/cluster_env.sh"
source "$TREE/optimization/setup_env_corona.sh"
source "$RUNTIME/runtime-cor.sh"
OWN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$OWN:$PYTHONPATH"
export LIVEFLOW_AGENT_FACTS_ONLY=1
export LIVEFLOW_WORKLOAD=dlio-pytorch LIVEFLOW_DLIO_HDF5_DECODE=1 LIVEFLOW_ANALYZER_FASTPATH=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$LIVEFLOW_AGENT_VENV_DIR/bin/python" -u "$OWN/replay-screen-node.py" "$@"

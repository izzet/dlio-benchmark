#!/bin/bash
set -euo pipefail
OWN=${DLIO_CAMPAIGN_CHECKOUT:?Set the pinned DLIO checkout}/examples/liveflow/cosmoflow_independent
[[ -d $OWN ]]
python3 -u "$OWN/forkserver-test.py"

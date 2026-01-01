#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sudo mn --custom topology/topo_4s_6h.py --topo fourswitch --controller=remote --switch ovs,protocols=OpenFlow13


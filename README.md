# Meta-S-MARL: Meta-Learning Enhanced Security-Aware MARL for SDN Congestion Control

Quick-start repo for the project:
- Mininet topology (topo/topo_4s_6h.py)
- Ryu controller skeleton (controller/Controller_UnifiedController.py)
- Log parser (tools/parse_logs.py)
- Agents:
  - Classifier (agents/classifier/q_learning_classifier.py)
  - Decision (agents/decision/dqn_agent.py)
- Training helpers and example scripts (training/)
- Run scripts (run/)

## Prerequisites
- Ubuntu (22.04 / 24.04) recommended
- Mininet installed: `sudo apt install mininet` or from source
- Open vSwitch installed
- Python: create a venv (recommended) and install requirements

## Quick install (example)
```bash
cd ~/Documents/pjt1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

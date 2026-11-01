"""Move intermediate files between Kaggle notebooks through a PRIVATE HF dataset repo.
Usage:  python sync.py push data      python sync.py pull ckpt
"""
import sys
from common import hub_push, hub_pull

cmd, rel = sys.argv[1], sys.argv[2]
{"push": hub_push, "pull": hub_pull}[cmd](rel)

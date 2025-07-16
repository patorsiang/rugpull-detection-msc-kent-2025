from collections import Counter
from pathlib import Path
import pandas as pd

import sys
sys.path.append(str(Path.cwd().parents[1]))
from scripts.utils import load_bytecode

def byte_freq_from_file(hex_file):
    bytecode = load_bytecode(hex_file)
    if bytecode.startswith("0x"):
        bytecode = bytecode[2:]

    byte_list = [bytecode[i:i+2] for i in range(0, len(bytecode), 2)]
    return dict(Counter(byte_list))

def get_byte_freq_from_files(files):
    records = []
    for file in files:
        freq = byte_freq_from_file(file)
        freq['address'] = file.stem.lower()
        records.append(freq)

    return pd.DataFrame(records).fillna(0).set_index('address')

import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import sys
sys.path.append(str(Path.cwd().parents[1]))
from scripts.utils import load_bytecode, get_opcodes, get_raw_opcode

def get_opcodes_freq_from_files(files):
    records = []
    for file in tqdm(files):
        bytecode = load_bytecode(file)
        raw_opcode = get_raw_opcode(bytecode)
        opcodes = dict(Counter(get_opcodes(raw_opcode)))
        opcodes['address'] = file.stem.lower()
        records.append(dict(Counter(opcodes)))
    return pd.DataFrame(records).fillna(0).set_index('address')

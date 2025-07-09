import pandas as pd
from evmdasm import EvmBytecode
from collections import Counter

def extract_opcodes(hex_code):
    try:
        evm = EvmBytecode(bytecode=hex_code)
        return [op.name if op.name else f"INVALID_0x{op.opcode:02x}" for op in evm.disassemble()]
    except Exception:
        return []

def get_opcode_freq(opcode_list):
    return Counter(opcode_list)

def build_feature_df(opcode_counter_list, address_list):
    # Union of all opcode keys
    all_features = sorted(set().union(*[c.keys() for c in opcode_counter_list]))
    aligned_data = []

    for addr, counter in zip(address_list, opcode_counter_list):
        row = {feature: counter.get(feature, 0) for feature in all_features}
        row["address"] = addr
        aligned_data.append(row)

    return pd.DataFrame(aligned_data)

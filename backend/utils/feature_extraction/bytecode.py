import os

from typing import List

from evmdasm import EvmBytecode
from collections import Counter
from scipy.stats import entropy
from backend.utils.feature_extraction.evm_cfg_builder.evm_cfg_builder.cfg.cfg import CFG
from backend.utils.feature_extraction.graph import extract_control_flow_graph_features

def load_bytecode(hex_file):
    with open(hex_file, 'r') as f:
        bytecode = f.read().strip()
    return bytecode

def get_byte_frequency_n_entropy(bytecode_hex: str):
    bytes_ = [bytecode_hex[i:i+2] for i in range(0, len(bytecode_hex), 2)]
    freq = Counter(bytes_)
    total = sum(freq.values())
    probs = [v / total for v in freq.values()]
    return {f'byte_{b}': freq[b] / total for b in freq}, entropy(probs, base=2)

def get_opcode_frequency_n_entropy(opcodes: List[str]):
    freq = Counter(opcodes)
    total = sum(freq.values())
    probs = [v / total for v in freq.values()]
    return entropy(probs, base=2)

def extract_graph_features(bytecode_hex: str):
    if bytecode_hex == '0x':
        return extract_control_flow_graph_features()
    return extract_control_flow_graph_features(CFG(bytecode_hex))

def extract_bytecode_features(hex_str: str):
    # Read hex file
    bytecode_hex = load_bytecode(hex_str)

    # Disassemble to opcodes
    bytecode = EvmBytecode(bytecode=bytecode_hex)
    opcodes = [op.name for op in bytecode.disassemble()]
    opcode_sequence = ' '.join(opcodes)

    byte_freq, byte_entropy = get_byte_frequency_n_entropy(bytecode_hex)
    opcode_entropy = get_opcode_frequency_n_entropy(opcodes)
    graph_feature = extract_graph_features(bytecode_hex)

    return {
        "opcode_sequence": opcode_sequence,
        "opcode_entropy": opcode_entropy,
        "opcode_count": len(opcodes),
        "unique_opcodes": len(set(opcodes)),
        "byte_entropy": byte_entropy,
        **byte_freq,
        **graph_feature
    }

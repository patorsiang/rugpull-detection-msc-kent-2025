import re
from evmdasm import EvmBytecode

def load_bytecode(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

def get_opcodes(opcodes):
    grouping_opcodes = []
    for opcode in opcodes:
        # Extract the first alphanumeric-only word (no digits, no special characters)
        match = re.match(r'^[a-zA-Z]+', opcode)

        if match:
            opcode_group = match.group()
            grouping_opcodes.append(opcode_group)
        else:
            grouping_opcodes.append(opcode)
    return grouping_opcodes

def get_raw_opcode(bytecode):
    evm_code = EvmBytecode(bytecode)
    return [instr.name for instr in evm_code.disassemble()]

def get_grouping_opcode_sequence(opcodes):
    grouping_opcode = get_opcodes(opcodes)
    return ' '.join(grouping_opcode)


def get_opcode_sequence(bytecode):
    return get_grouping_opcode_sequence(get_raw_opcode(bytecode))

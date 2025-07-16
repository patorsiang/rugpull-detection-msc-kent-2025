import re
from evmdasm import EvmBytecode

def load_bytecode(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

def get_grouping_opcode_sequence(opcodes):
    grouping_opcode = []
    for opcode in opcodes:
        match = re.match(r'^[a-zA-Z]+', opcode)

        if match:
            opcode_group = match.group()
            grouping_opcode.append(opcode_group)
        else:
            grouping_opcode.append(opcode)
    return ' '.join(opcodes)


def get_opcode_sequence(bytecode):
    evm_code = EvmBytecode(bytecode)
    return get_grouping_opcode_sequence([instr.name for instr in evm_code.disassemble()])

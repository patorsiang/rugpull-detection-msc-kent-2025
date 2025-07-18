import re
from evmdasm import EvmBytecode
from sklearn.metrics import f1_score
import numpy as np

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

def tune_thresholds(y_true, y_pred_prob, metric='f1'):
    y_true = np.asarray(y_true)          # Fix: convert to NumPy
    y_pred_prob = np.asarray(y_pred_prob)

    best_thresholds = []
    best_scores = []

    for i in range(y_true.shape[1]):
        label_true = y_true[:, i]
        label_probs = y_pred_prob[:, i]  # Fix here too

        thresholds = np.linspace(0.0, 1.0, 101)
        scores = []

        for t in thresholds:
            label_pred = (label_probs >= t).astype(int)
            if metric == 'f1':
                score = f1_score(label_true, label_pred, zero_division=0)
            scores.append(score)

        best_t = thresholds[np.argmax(scores)]
        best_score = np.max(scores)

        best_thresholds.append(best_t)
        best_scores.append(best_score)

        print(f"Label {i}: Best threshold = {best_t:.2f}, Best {metric} = {best_score:.4f}")

    return best_thresholds, best_scores

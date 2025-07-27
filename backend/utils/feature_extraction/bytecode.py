import os
from pathlib import Path
import pandas as pd
from evmdasm import EvmBytecode
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
from collections import Counter
import joblib

# === Helper Functions ===

def load_bytecode(hex_file):
    with open(hex_file, 'r') as f:
        bytecode = f.read().strip()
    return bytecode

def get_opcode_entropy(opcodes):
    freqs = Counter(opcodes)
    total = sum(freqs.values())
    probs = [v / total for v in freqs.values()]
    return entropy(probs, base=2)

def get_byte_entropy(bytecode_hex):
    bytes_ = [bytecode_hex[i:i+2] for i in range(0, len(bytecode_hex), 2)]
    freqs = Counter(bytes_)
    total = sum(freqs.values())
    probs = [v / total for v in freqs.values()]
    return entropy(probs, base=2)

def get_byte_frequency(bytecode_hex):
    bytes_ = [bytecode_hex[i:i+2] for i in range(0, len(bytecode_hex), 2)]
    freq = Counter(bytes_)
    total = sum(freq.values())
    return {f'byte_{b}': freq[b] / total for b in freq}

# === Main Feature Extractor ===

def extract_bytecode_static_features(hex_path):
    # Read hex file
    bytecode_hex = load_bytecode(hex_path)

    # Disassemble to opcodes
    bytecode = EvmBytecode(bytecode=bytecode_hex)
    opcodes = [op.name for op in bytecode.disassemble()]
    opcode_sequence = ' '.join(opcodes)

    # Extract features
    opcode_entropy = get_opcode_entropy(opcodes)
    byte_freq = get_byte_frequency(bytecode_hex)
    byte_entropy = get_byte_entropy(bytecode_hex)

    return {
        "Address": os.path.basename(hex_path).replace(".hex", ""),
        "opcode_sequence": opcode_sequence,
        "opcode_entropy": opcode_entropy,
        "byte_entropy": byte_entropy,
        **byte_freq  # merge byte frequency into the result
    }

def generate_ngram_features(opcode_sequences, MODEL_PATH, ngram_range=(1, 3), max_features=1000, min_df=2, use_saved_model=False):
    vectorizer_path = os.path.join(MODEL_PATH, 'opcode_vectorizer.pkl')
    print(vectorizer_path)
    if use_saved_model and os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        X = vectorizer.transform(opcode_sequences)
    else:
        vectorizer = CountVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df
        )
        X = vectorizer.fit_transform(opcode_sequences)
        # Save the fitted vectorizer
        os.makedirs(MODEL_PATH, exist_ok=True)
        joblib.dump(vectorizer, vectorizer_path)

    feature_names = vectorizer.vocabulary_.keys()
    return pd.DataFrame(X.toarray(), columns=feature_names), vectorizer


def build_bytecode_feature_dataframe(hex_dir, MODEL_PATH):
    # Step 1: Static extraction
    feature_rows = []
    for hex_file in list(Path(hex_dir).glob("*.hex")):
        row = extract_bytecode_static_features(hex_file)
        feature_rows.append(row)

    df_static = pd.DataFrame(feature_rows)

    # Step 2: N-gram vectorization
    ngram_df, vectorizer = generate_ngram_features(df_static["opcode_sequence"].tolist(), MODEL_PATH)

    # Step 3: Merge
    df_final = pd.concat([df_static.drop(columns=["opcode_sequence"]), ngram_df], axis=1).set_index('Address').fillna(0)

    return df_final, vectorizer

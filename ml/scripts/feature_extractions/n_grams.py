import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

import sys
sys.path.append(str(Path.cwd().parents[1]))
from scripts.utils import load_bytecode, get_opcode_sequence

def get_n_grams(files, start=2, end=3):
    records = []

    for file in files:
        freq = dict()
        bytecode = load_bytecode(file)
        freq['opcode_sequence'] = get_opcode_sequence(bytecode)
        freq['address'] = file.stem.lower()
        records.append(freq)

    seq_df = pd.DataFrame(records).fillna(0).set_index('address')

    vectorizer = CountVectorizer(ngram_range=(start, end), analyzer='word', max_features=1000)
    X_ngrams = vectorizer.fit_transform(seq_df['opcode_sequence'])

    X = pd.DataFrame(X_ngrams.toarray(), columns=vectorizer.get_feature_names_out())
    X['address'] = seq_df.index
    X = X.set_index('address')

    return X

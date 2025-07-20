import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path.cwd().parents[1]))
from scripts.utils import load_bytecode, get_opcode_sequence

def get_n_grams_from_files(files, ngram_range=(2, 3)):
    records = []

    for file in tqdm(files):
        freq = dict()
        bytecode = load_bytecode(file)
        freq['opcode_sequence'] = get_opcode_sequence(bytecode)
        freq['address'] = file.stem.lower()
        records.append(freq)

    seq_df = pd.DataFrame(records).fillna(0).set_index('address')

    vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer='word', max_features=1000)
    X_ngrams = vectorizer.fit_transform(seq_df['opcode_sequence'])

    X = pd.DataFrame(X_ngrams.toarray(), columns=vectorizer.get_feature_names_out())
    X['address'] = seq_df.index
    X = X.set_index('address')

    return X

def extract_n_grams_for_unlabeled(files, feature_cols, ngram_range=(2, 3)):

    document = []

    for file in tqdm(files):
        bytecode = load_bytecode(file)
        content = get_opcode_sequence(bytecode)
        document.append(content)

    vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer='word', vocabulary=feature_cols)

    X = vectorizer.fit_transform(document)

    return X

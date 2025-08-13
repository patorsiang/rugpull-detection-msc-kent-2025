import os
import pandas as pd
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = '../models'

def generate_tf_idf_features(documents, MODEL_PATH, max_features=2000, min_df=2, use_saved_model=False):
    vectorizer_path = os.path.join(MODEL_PATH, 'tf_idf_vectorizer.pkl')

    if use_saved_model and os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        X = vectorizer.transform(documents)
    else:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer='word',
            token_pattern=r'\b\w+\b',
            max_features=max_features,
            min_df=min_df
        )
        X = vectorizer.fit_transform(documents)
        # Save the fitted vectorizer
        os.makedirs(MODEL_PATH, exist_ok=True)
        joblib.dump(vectorizer, vectorizer_path)

    feature_names = vectorizer.vocabulary_.keys()
    return pd.DataFrame(X.toarray(), columns=feature_names), vectorizer

def build_sol_feature_dataframe(sol_dir, MODEL_PATH, max_features=2000, min_df=2, use_saved_model=False, address=None):
    documents_sol = []
    addresses_sol = []

    for file in list(Path(sol_dir).glob(f"{address if address is not None else '*'}.sol")):
        address = file.stem.lower()
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            documents_sol.append(content)
            addresses_sol.append({'Address': address})

    if not documents_sol:
        print(f"[WARN] No .sol files found for address={address} in {sol_dir}")
        return pd.DataFrame(), None

    df_static = pd.DataFrame(addresses_sol)

    # Step 2: N-gram vectorization
    tf_idf_df, vectorizer = generate_tf_idf_features(documents_sol, MODEL_PATH, max_features, min_df, use_saved_model)

    # Step 3: Merge
    df_final = pd.concat([df_static, tf_idf_df], axis=1).set_index('Address').fillna(0)

    return df_final, vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def get_tf_idf_vector(files, max_feature=2000):
    documents = []

    for file in tqdm(files):
         with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            documents.append(content)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer='word',
        token_pattern=r'\b\w+\b',
        max_features=max_feature
    )

    X = vectorizer.fit_transform(documents)

    return X, vectorizer.vocabulary_

from sklearn.feature_extraction.text import TfidfVectorizer

def get_tf_idf_vector(files):
    documents = []

    for file in files:
         with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            documents.append(content)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer='word',
        token_pattern=r'\b\w+\b',
        max_features=10000
    )

    X = vectorizer.fit_transform(documents)

    return X, vectorizer.vocabulary_

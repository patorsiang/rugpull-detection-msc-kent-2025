# Feature Extraction Methods

## From Bytecode

| **Feature**                        | **How to Extract**                                                                    | **Data Format**                 | **Recommended ML Models**                          |
| ---------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------- | -------------------------------------------------- |
| 🔹 **Opcode Frequency**            | Disassemble `.hex` using `evmdasm`, count opcode frequencies                          | Tabular (opcode counts per row) | Logistic Regression, Random Forest, XGBoost, MLP   |
| 🔹 **Opcode N-grams**              | Tokenize opcodes, extract n-grams using `CountVectorizer` (bigram/trigram)            | Sparse matrix                   | SVM, Logistic Regression, Naive Bayes, XGBoost     |
| 🔹 **Opcode Entropy**              | Use `scipy.stats.entropy` on opcode distribution per contract                         | Scalar per contract             | Thresholding, MLP, XGBoost                         |
| 🔹 **Byte Entropy / Frequency**    | Convert `.hex` to bytes; compute byte histogram or entropy                            | Tabular or vector               | Logistic Regression, MLP, XGBoost                  |
| 🔹 **ByteTransformer Embeddings**  | Run ByteTransformer model (requires PyTorch & pretrained weights)                     | Dense vector (512+)             | MLP, k-NN, SVM, clustering, anomaly detection      |
| 🔹 **Control Flow Graph Features** | Use `Gigahorse`, `Ethervm.io`, or EVM decompilers to extract control/data flow graphs | Graph / vector summary          | GNN, Graph2Vec, or extract graph stats for XGBoost |

## From Source Code

| **Source Code Features**      | **How to Extract**                                                          | **Data Format**              | **Recommended ML Models**                             |
| ----------------------------- | --------------------------------------------------------------------------- | ---------------------------- | ----------------------------------------------------- |
| 🔹 **AST Feature Extraction** | Run `solc --ast-json`; extract tree patterns, depth, node types             | Structured JSON → vectorized | Random Forest, XGBoost, Tree-based models             |
| 🔹 **Function-Level Stats**   | Parse AST or use Slither to count function types, modifiers, visibility     | Tabular                      | Random Forest, XGBoost, Logistic Regression           |
| 🔹 **CodeBERT Embeddings**    | Use HuggingFace's CodeBERT (`microsoft/codebert-base`); tokenize `.sol`     | Dense vector                 | MLP, SVM, fine-tune with transformer head             |
| 🔹 **TF-IDF / BoW**           | `TfidfVectorizer` from sklearn on tokenized `.sol` or `.hex`                | Sparse matrix                | Logistic Regression, SVM, Naive Bayes                 |
| 🔹 **Pattern-Based Rules**    | Regex or AST pattern match for suspicious functions (e.g., `mint()` access) | Tabular flags (0/1)          | Rule-based + ML hybrid (XGBoost, MLP)                 |
| 🔹 **Slither Static Reports** | Run `slither some.sol --json slither-out.json`; flatten output              | Tabular or JSON              | Logistic Regression, Random Forest, for pre-screening |

## Other Advanced Options

| **Advanced**                     | **How to Extract**                                                          | **Data Format**       | **Recommended ML Models**                |
| -------------------------------- | --------------------------------------------------------------------------- | --------------------- | ---------------------------------------- |
| 🔹 **Source Code Metrics**       | Use `radon`, `lizard`, `code2vec` for complexity, Halstead, etc.            | Tabular               | Logistic Regression, XGBoost, Clustering |
| 🔹 **Decompilation Features**    | Use `Ethersplay`, `Panoramix`, or `Porosity` for higher-level code          | Tabular / Tree        | MLP, Tree-based, Rule models             |
| 🔹 **Taint Analysis**            | Run `slither` or `mythril` to detect function-call chains with taint issues | Tabular               | Binary classifiers, Feature-based models |
| 🔹 **Symbolic Execution**        | Use `Mythril` or `Manticore` to simulate execution paths                    | Path summary features | Binary classifiers (vulnerable or not)   |
| 🔹 **Pretraining + Fine-tuning** | Fine-tune CodeBERT/GraphCodeBERT on labeled contracts                       | Embedding + task head | Transformers, MLP on pooled output       |

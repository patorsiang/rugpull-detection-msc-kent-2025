# Test Models with CRPWwarner Large Sample

## Mint

| Extraction | Model | Accuracy (f1-score) |
| - | - | - |
| Graph Statistic | MPL 512 \* 256 \* 128 (Multi) | 0.85 |
| Byte Frequency | MPL 512 \* 256 \* 128 (Multi) | 0.39 |
| Opcode Frequency | MultiOutputClassifier(AdaBoostClassifier()) (Multi) | 0.24 |
| TF-IDF | DecisionTreeClassifier (mint-model-f1-1.000000.pkl) | 0.16 |
| Opcode Frequency | MultiOutputClassifier(XGBClassifier()) | 0.14 |

## Leak

| Extraction | Model | Accuracy (f1-score) |
| - | - | - |
| Byte Frequency | MPL 512 \* 256 \* 128 (Multi) | 0.28 |
| Graph Statistic | MPL 512 \* 256 \* 128 (Multi) | 0.25 |
| Opcode Frequency | MultiOutputClassifier(AdaBoostClassifier()) (Multi) | 0.23 |
| Byte Frequency | GaussianNB (leak-model-f1-0.878261.pkl) | 0.20 |
| Opcode Frequency | MultiOutputClassifier(XGBClassifier()) | 0.15 |

## Limit

| Extraction | Model | Accuracy (f1-score) |
| - | - | - |
| Byte Frequency | MPL 512 \* 256 \* 128 (Multi) | 0.85 |
| Graph Statistic | MPL 512 \* 256 \* 128 (Multi) | 0.46 |
| n-grams from gigahorse | LogisticRegression (limit-model-f1-0.928205.pkl) | 0.19 |
| Opcode Frequency | MultiOutputClassifier(AdaBoostClassifier()) (Multi) | 0.12 |
| Opcode Frequency | MultiOutputClassifier(XGBClassifier()) | 0.22 |

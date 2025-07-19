# Result

## Mint

| Feature Extraction | Model | Precision | Recall | F1-score |
| - | - | - | - | - |
| Opcode Frequency | KNeighborsClassifier() | 0.619565 | 0.613636 | 0.615873 |
| Byte Frequency | DecisionTreeClassifier() | 0.605769 | 0.613636 | 0.607143 |
| TF-IDF (Source Code) | KNeighborsClassifier() | 0.681818 | 0.681818 | 0.606061 |
| N-Grams | XGBClassifier() | 0.601103 | 0.613636 | 0.592593 |
| Graph Statistic | AdaBoostClassifier() | 0.590909 | 0.590909 | 0.590909 |
| TF-IDF (OPCODE) | MPL (512x256x128) | 0.68 | 0.57 | 0.55 |
| Control Flow Graph | GCN | 0.28 | 0.50 | 0.36 |

## Leak

| Feature Extraction | Model | Precision | Recall | F1-score |
| - | - | - | - | - |
| TF-IDF (Source Code) | XGBClassifier() | 0.738866 | 0.734127 | 0.718475 |
| Graph Statistic | MPL (512x256x128) | 0.82 | 0.64 | 0.61 |
| Opcode Frequency | MPL (512x256x128) | 0.81 | 0.61 | 0.56 |
| Byte Frequency | XGBClassifier() | 0.555556 | 0.555556 | 0.555556 |
| N-Grams | SGDClassifier() | 0.520243 | 0.519841 | 0.519520 |
| Control Flow Graph | GCN | 0.39 | 0.45 | 0.38 |
| TF-IDF (OPCODE) | MPL (512x256x128) | 0.28 | 0.50 | 0.36 |

## Limit

| Feature Extraction | Model | Precision | Recall | F1-score |
| - | - | - | - | - |
| Opcode Frequency | ExtraTreesClassifier() | 0.756410 | 0.708333 | 0.725714 |
| N-Grams | XGBClassifier() | 0.705714 | 0.687500 | 0.695238 |
| Control Flow Graph | GCN | 0.90 | 0.62 | 0.64 |
| Graph Statistic | MLPClassifier() | 0.642857 | 0.583333 | 0.589744 |
| Byte Frequency |  MPL (512x256x128) | 0.73 | 0.60 | 0.62 |
| TF-IDF (Source Code) | MPL (512x256x128) | 0.79 | 0.67 | 0.69 |
| TF-IDF (OPCODE) | MPL (512x256x128) | 0.38 | 0.50 | 0.43 |

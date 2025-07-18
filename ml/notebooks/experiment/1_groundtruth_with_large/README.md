# Result

## Mint

| Feature Extraction | Model | Precision | Recall | F1-score |
| - | - | - | - | - |
| Byte Frequency | DecisionTreeClassifier() | 0.630952 | 0.636364 | 0.632963 |
| TF-IDF (Source Code) | KNeighborsClassifier() | 0.681818 | 0.681818 | 0.606061 |
| Opcode Frequency | KNeighborsClassifier() | 0.619565 | 0.613636 | 0.615873 |
| N-Grams | XGBClassifier() | 0.601103 | 0.613636 | 0.592593 |
| Graph Statistic | AdaBoostClassifier() | 0.590909 | 0.590909 | 0.590909 |
| TF-IDF (OPCODE) | LogisticRegression() | 0.333333 | 0.5 | 0.40 |
| Control Flow Graph | GCN | 0.28 | 0.50 | 0.36 |

## Leak

| Feature Extraction | Model | Precision | Recall | F1-score |
| - | - | - | - | - |
| TF-IDF (Source Code) | XGBClassifier() | 0.738866 | 0.734127 | 0.718475 |
| Graph Statistic | AdaBoostClassifier() | 0.598039 | 0.599206 | 0.593353 |
| Opcode Frequency | KNeighborsClassifier() | 0.571429 | 0.571429 | 0.562500 |
| Byte Frequency | MPL (512x256x128) | 0.555556 | 0.555556 | 0.555556 |
| TF-IDF (OPCODE) | MPL (512x256x128) | 0.68 | 0.58 | 0.54 |
| N-Grams | GaussianNB() | 0.597143| 0.567460 | 0.507692 |
| Control Flow Graph | GCN | 0.39 | 0.45 | 0.38 |

## Limit

| Feature Extraction | Model | Precision | Recall | F1-score |
| - | - | - | - | - |
| Opcode Frequency | RandomForestClassifier() | 0.666667 | 0.666667 | 0.666667 |
| N-Grams | XGBClassifier() | 0.705714 | 0.687500 | 0.695238 |
| Control Flow Graph | GCN | 0.90 | 0.62 | 0.64 |
| Graph Statistic | XGBClassifier() | 0.90 | 0.62 | 0.64 |
| Byte Frequency | ExtraTreesClassifier() | 0.653846 | 0.625000 | 0.634286 |
| TF-IDF (Source Code) | RandomForestClassifier() | 0.522857 | 0.520833 | 0.521088 |
| TF-IDF (OPCODE) | MPL (512x256x128) | 0.38 | 0.50 | 0.43 |

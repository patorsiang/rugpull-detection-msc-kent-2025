# Baseline Model Comparative table (groundtruth 69 records)

## Note

for feature extraction from gigahorse, using 7 mins per contracts to get IR.

## Byte Frequency

### for multi-labels

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MPL 512 \* 256 \* 128 | 0.75 | 0.89 | 0.78 |
| OneVsRestClassifier(AdaBoostClassifier()) | 0.952381 | 0.611111 | 0.7388891 |
| MultiOutputClassifier(GaussianNB()) | 0.768519 | 0.574074 | 0.6518522 |
| OneVsRestClassifier(GaussianNB()) | 0.768519 | 0.574074 | 0.6518523 |
| MultiOutputClassifier(XGBClassifier()) | 0.933333 | 0.425926 | 0.5793654 |
| OneVsRestClassifier(XGBClassifier()) | 0.933333 | 0.425926 | 0.579365 |

### for binary

#### Mint

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| AdaBoostClassifier() | 0.863636 | 0.750000 | 0.7543861 |
| XGBClassifier() | 0.863636 | 0.750000 | 0.7543862 |
| SVC() | 0.863636 | 0.750000 | 0.7543863 |
| KNeighborsClassifier() | 0.725000 | 0.687500 | 0.6888894 |
| GaussianNB() | 0.725000 | 0.687500 | 0.688889 |

#### Leak

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| GaussianNB() | 0.958333 | 0.833333 | 0.8782611 |
| LGBMClassifier() | 0.958333 | 0.833333 | 0.8782612 |
| XGBClassifier() | 0.923077 | 0.666667 | 0.7083333 |
| AdaBoostClassifier() | 0.666667 | 0.621212 | 0.6347834 |
| RandomForestClassifier() | 0.392857 | 0.500000 | 0.440000 |

#### Limit

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| AdaBoostClassifier() | 0.714286 | 0.733333 | 0.7083331 |
| KNeighborsClassifier() | 0.600000 | 0.588889 | 0.5906432 |
| XGBClassifier() | 0.622222 | 0.622222 | 0.5714293 |
| LGBMClassifier() | 0.622222 | 0.622222 | 0.5714294 |
| LogisticRegression() | 0.571429 | 0.577778 | 0.562500 |

## GCN (from evm_cfg_builder)

| | precision | recall | f1-score | support |
| - | - | - | - | - |
| mint | 0.71 | 0.83 | 0.77 | 6 |
| leak | 0.22 | 1.00 | 0.36 | 2 |
| limit | 0.62 | 0.83 | 0.71 | 6 |
| | | | | |
| micro avg | 0.50 | 0.86 | 0.63 | 14 |
| macro avg | 0.52 | 0.89 | 0.62 | 14 |
| weighted avg | 0.61 | 0.86 | 0.69 | 14 |
| samples avg | 0.48 | 0.62 | 0.52 | 14 |

### For binary (mint)

| | precision | recall | f1-score | support |
| - | - | - | - | - |
| 0.0 | 1.00 | 0.38 | 0.55 | 8 |
| 1.0 | 0.55 | 1.00 | 0.71 | 6 |
| | | | | |
| accuracy | | | 0.64 | 14 |
| macro avg | 0.77 | 0.69 | 0.63 | 14 |
| weighted avg | 0.81 | 0.64 | 0.61 | 14 |

## GCN (from IR gigahorse)

| | precision | recall | f1-score | support |
| - | - | - | - | - |
| mint | 0.00 | 0.00 | 0.00 | 4 |
| leak | 0.00 | 0.00 | 0.00 | 2 |
| limit | 0.00 | 0.00 | 0.00 | 5 |
| | | | | |
| micro avg | 0.00 | 0.00 | 0.00 | 11 |
| macro avg | 0.00 | 0.00 | 0.00 | 11 |
| weighted avg | 0.00 | 0.00 | 0.00 | 11 |
| samples avg | 0.00 | 0.00 | 0.00 | 11 |

### For binary (mint) - IR gigahorse

| | precision | recall | f1-score | support |
| - | - | - | - | - |
| 0.0 | 0.69 | 0.90 | 0.78 | 10 |
| 1.0 | 0.00 | 0.00 | 0.00 | 4 |
| | | | | |
| accuracy | | | 0.64 | 14 |
| macro avg | 0.35 | 0.45 | 0.39 | 14 |
| weighted avg | 0.49 | 0.64 | 0.56 | 14 |

## Graphs Statistic

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MPL 512 \* 256 \* 128 | 0.89 | 0.89 | 0.76 |
| MultiOutputClassifier(SGDClassifier()) | 0.547619 | 0.666667 | 0.5846791 |
| MultiOutputClassifier(KNeighborsClassifier()) | 0.583333 | 0.444444 | 0.5019612 |
| OneVsRestClassifier(KNeighborsClassifier()) | 0.583333 | 0.444444 | 0.5019613 |
| MultiOutputClassifier(DecisionTreeClassifier()) | 0.571429 | 0.462963 | 0.4945054 |
| OneVsRestClassifier(RandomForestClassifier()) | 0.571429 | 0.462963 | 0.494505 |

### for binary -- Graphs Statistic

#### Mint -- Graphs Statistic

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| KNeighborsClassifier() | 0.900000 | 0.833333 | 0.8444441 |
| DecisionTreeClassifier() | 0.785714 | 0.791667 | 0.7846152 |
| AdaBoostClassifier() | 0.785714 | 0.791667 | 0.7846153 |
| RandomForestClassifier() | 0.785714 | 0.791667 | 0.7846154 |
| XGBClassifier() | 0.708333 | 0.708333 | 0.708333 |

#### Leak -- Graphs Statistic

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| DecisionTreeClassifier() | 0.666667 | 0.621212 | 0.6347831 |
| LGBMClassifier() | 0.666667 | 0.621212 | 0.6347832 |
| RandomForestClassifier() | 0.392857 | 0.5000000.4400003 |
| LogisticRegression() | 0.392857 | 0.500000 | 0.4400004 |
| MLPClassifier() | 0.392857 | 0.500000 | 0.440000 |

#### Limit -- Graphs Statistic

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| DecisionTreeClassifier() | 0.777778 | 0.777778 | 0.7142861 |
| RandomForestClassifier() | 0.777778 | 0.777778 | 0.7142862 |
| AdaBoostClassifier() | 0.777778 | 0.777778 | 0.7142863 |
| ExtraTreesClassifier() | 0.777778 | 0.777778 | 0.7142864 |
| XGBClassifier() | 0.666667 | 0.677778 | 0.641026 |

## Graphs Statistic -- Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| OneVsRestClassifier(GaussianNB()) | 0.465608 | 0.690476 | 0.5052911 |
| MultiOutputClassifier(GaussianNB()) | 0.465608 | 0.690476 | 0.5052912 |
| MultiOutputClassifier(XGBClassifier()) | 0.428571 | 0.559524 | 0.4131053 |
| OneVsRestClassifier(XGBClassifier()) | 0.428571 | 0.559524 | 0.4131054 |
| OneVsRestClassifier(AdaBoostClassifier()) | 0.312169 | 0.571429 | 0.359307 |

### for binary -- Graphs Statistic -- Gigahorse

#### Mint -- Graphs Statistic -- Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| GaussianNB() | 0.642857 | 0.791667 | 0.5906431 |
| XGBClassifier() | 0.642857 | 0.791667 | 0.590643 |
| SVC() | 0.625000 | 0.750000 | 0.533333 |
| DecisionTreeClassifier() | 0.611111 | 0.708333 | 0.475936 |
| AdaBoostClassifier() | 0.611111 | 0.708333 | 0.475936 |

#### Leak -- Graphs Statistic -- Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| GaussianNB() | 0.742424 | 0.700 | 0.7142861 |
| LogisticRegression() | 0.625000 | 0.575 | 0.5757582 |
| XGBClassifier() | 0.625000 | 0.575 | 0.5757583 |
| SVC() | 0.357143 | 0.500 | 0.4166674 |
| MLPClassifier() | 0.357143 | 0.500 | 0.416667 |

#### Limit -- Graphs Statistic -- Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| DecisionTreeClassifier() | 0.714286 | 0.714286 | 0.714286 |
| AdaBoostClassifier() | 0.714286 | 0.714286 | 0.714286 |
| ExtraTreesClassifier() | 0.714286 | 0.714286 | 0.714286|
| RandomForestClassifier() | 0.645833 | 0.642857 | 0.641026 |
| KNeighborsClassifier() | 0.500000 | 0.500000 | 0.497436 |

## Opcode Entropy

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MPL 512 \* 256 \* 128 | 0.44 | 1.00 | 0.59 |
| MultiOutputClassifier(XGBClassifier()) | 0.533333 | 0.277778 | 0.3484851 |
| OneVsRestClassifier(XGBClassifier()) | 0.533333 | 0.277778 | 0.3484852 |
| MultiOutputClassifier(ExtraTreesClassifier()) | 0.261905 | 0.185185 | 0.2051283 |
| MultiOutputClassifier(DecisionTreeClassifier()) | 0.261905 | 0.185185 | 0.2051284 |
| OneVsRestClassifier(DecisionTreeClassifier()) | 0.261905 | 0.185185 | 0.205128 |

### for binary -- Opcode Entropy

#### Mint -- Opcode Entropy

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| XGBClassifier() | 0.633333 | 0.625000 | 0.6256681 |
| LogisticRegression() | 0.285714 | 0.500000 | 0.3636362 |
| LGBMClassifier() | 0.285714 | 0.500000 | 0.3636363 |
| SVC() | 0.285714 | 0.500000 | 0.3636364 |
| MLPClassifier() | 0.285714 | 0.500000 | 0.363636 |

#### Leak -- Opcode Entropy

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| LogisticRegression() | 0.392857 | 0.5 | 0.441 |
| DecisionTreeClassifier() | 0.392857 | 0.5 | 0.442 |
| RandomForestClassifier() | 0.392857 | 0.5 | 0.443 |
| AdaBoostClassifier() | 0.392857 | 0.5 | 0.444 |
| ExtraTreesClassifier() | 0.392857 | 0.5 | 0.44 |

#### Limit -- Opcode Entropy

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| XGBClassifier() | 0.727273 | 0.666667 | 0.5625001 |
| SVC() | 0.575000 | 0.566667 | 0.4974362 |
| KNeighborsClassifier() | 0.575000 | 0.566667 | 0.4974363 |
| LGBMClassifier() | 0.428571 | 0.422222 | 0.4166674 |
| SGDClassifier() | 0.321429 | 0.500000 | 0.391304 |

## Opcode Frequency

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| OneVsRestClassifier(KNeighborsClassifier()) | 0.866667 | 0.500000 | 0.5994151 |
| MultiOutputClassifier(KNeighborsClassifier()) | 0.866667 | 0.500000 | 0.5994152 |
| MPL 512 \* 256 \* 128 | 0.43 | 1.00 | 0.58 |
| OneVsRestClassifier(AdaBoostClassifier()) | 0.722222 | 0.370370 | 0.4777783 |
| MultiOutputClassifier(GaussianNB()) | 0.421645 | 0.537037 | 0.4705134 |
| OneVsRestClassifier(GaussianNB()) | 0.421645 | 0.537037 | 0.470513 |

### for binary -- Opcode Frequency

#### Mint -- Opcode Frequency

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MPL 512 \* 256 \* 128 | 0.94 | 0.92 | 0.93 |
| KNeighborsClassifier() | 0.863636 | 0.750000 | 0.754386 |
| DecisionTreeClassifier() | 0.833333 | 0.666667 | 0.650000 |
| LGBMClassifier() | 0.833333 | 0.666667 | 0.650000 |
| AdaBoostClassifier() | 0.833333 | 0.666667 | 0.650000 |
| SGDClassifier() | 0.633333 | 0.625000| 0.625668 |

#### Leak -- Opcode Frequency

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| KNeighborsClassifier() | 0.923077 | 0.666667 | 0.708333 |
| AdaBoostClassifier() | 0.666667 | 0.621212 | 0.634783 |
| GaussianNB() | 0.488889 | 0.484848 | 0.475000 |
| RandomForestClassifier() | 0.392857 | 0.500000 | 0.440000 |
| SGDClassifier() | 0.392857 | 0.500000 | 0.440000 |

#### Limit -- Opcode Frequency

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| LogisticRegression() | 0.666667 | 0.677778 | 0.641026 |
| DecisionTreeClassifier() | 0.666667 | 0.677778 | 0.641026 |
| MLPClassifier() | 0.600000 | 0.588889 | 0.590643 |
| RandomForestClassifier() | 0.571429 | 0.577778 | 0.5625004 |
| LGBMClassifier() | 0.571429 | 0.577778 | 0.562500 |

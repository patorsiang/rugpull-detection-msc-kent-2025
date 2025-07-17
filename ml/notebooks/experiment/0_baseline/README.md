# Baseline Model Comparative table (groundtruth 69 records)

- extract with models

## Note

for feature extraction from gigahorse, using 7 mins per contracts to get IR and report

## Byte Frequency

### Byte Frequency - Multi-Labels

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MPL 512 \* 256 \* 128 | 0.75 | 0.89 | 0.78 |
| OneVsRestClassifier(AdaBoostClassifier()) | 0.952381 | 0.611111 | 0.738889 |
| MultiOutputClassifier(GaussianNB()) | 0.768519 | 0.574074 | 0.651852 |
| OneVsRestClassifier(GaussianNB()) | 0.768519 | 0.574074 | 0.651852 |
| MultiOutputClassifier(XGBClassifier()) | 0.933333 | 0.425926 | 0.579365 |
| OneVsRestClassifier(XGBClassifier()) | 0.933333 | 0.425926 | 0.579365 |

### Byte Frequency - Binary

#### Mint - Byte Frequency - Binary

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| AdaBoostClassifier() | 0.863636 | 0.750000 | 0.754386 |
| XGBClassifier() | 0.863636 | 0.750000 | 0.754386 |
| SVC() | 0.863636 | 0.750000 | 0.754386 |
| KNeighborsClassifier() | 0.725000 | 0.687500 | 0.688889 |
| GaussianNB() | 0.725000 | 0.687500 | 0.688889 |

#### Leak - Byte Frequency - Binary

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| GaussianNB() | 0.958333 | 0.833333 | 0.878261 |
| LGBMClassifier() | 0.958333 | 0.833333 | 0.878261 |
| XGBClassifier() | 0.923077 | 0.666667 | 0.708333 |
| AdaBoostClassifier() | 0.666667 | 0.621212 | 0.634783 |
| RandomForestClassifier() | 0.392857 | 0.500000 | 0.440000 |

#### Limit - Byte Frequency - Binary

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| AdaBoostClassifier() | 0.714286 | 0.733333 | 0.708333 |
| KNeighborsClassifier() | 0.600000 | 0.588889 | 0.590643 |
| XGBClassifier() | 0.622222 | 0.622222 | 0.571429 |
| LGBMClassifier() | 0.622222 | 0.622222 | 0.571429 |
| LogisticRegression() | 0.571429 | 0.577778 | 0.562500 |

## GCN

### GCN (from evm_cfg_builder)

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

### GCN (from evm_cfg_builder) -- binary (mint)

| | precision | recall | f1-score | support |
| - | - | - | - | - |
| 0.0 | 1.00 | 0.38 | 0.55 | 8 |
| 1.0 | 0.55 | 1.00 | 0.71 | 6 |
| | | | | |
| accuracy | | | 0.64 | 14 |
| macro avg | 0.77 | 0.69 | 0.63 | 14 |
| weighted avg | 0.81 | 0.64 | 0.61 | 14 |

### GCN (from IR gigahorse)

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

### GCN (from IR gigahorse) -- binary (mint)

| | precision | recall | f1-score | support |
| - | - | - | - | - |
| 0.0 | 0.69 | 0.90 | 0.78 | 10 |
| 1.0 | 0.00 | 0.00 | 0.00 | 4 |
| | | | | |
| accuracy | | | 0.64 | 14 |
| macro avg | 0.35 | 0.45 | 0.39 | 14 |
| weighted avg | 0.49 | 0.64 | 0.56 | 14 |

## Graphs Statistic

### Graphs Statistic - Multi-Labels

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MPL 512 \* 256 \* 128 | 0.89 | 0.89 | 0.76 |
| MultiOutputClassifier(SGDClassifier()) | 0.547619 | 0.666667 | 0.584679 |
| MultiOutputClassifier(KNeighborsClassifier()) | 0.583333 | 0.444444 | 0.501961 |
| OneVsRestClassifier(KNeighborsClassifier()) | 0.583333 | 0.444444 | 0.501961 |
| MultiOutputClassifier(DecisionTreeClassifier()) | 0.571429 | 0.462963 | 0.494505 |
| OneVsRestClassifier(RandomForestClassifier()) | 0.571429 | 0.462963 | 0.494505 |

## Graphs Statistic - Binary

### Mint - Graphs Statistic - Binary

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| KNeighborsClassifier() | 0.900000 | 0.833333 | 0.844444 |
| DecisionTreeClassifier() | 0.785714 | 0.791667 | 0.784615 |
| AdaBoostClassifier() | 0.785714 | 0.791667 | 0.784615 |
| RandomForestClassifier() | 0.785714 | 0.791667 | 0.784615 |
| XGBClassifier() | 0.708333 | 0.708333 | 0.708333 |

### Leak - Graphs Statistic - Binary

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| DecisionTreeClassifier() | 0.666667 | 0.621212 | 0.634783 |
| LGBMClassifier() | 0.666667 | 0.621212 | 0.634783 |
| RandomForestClassifier() | 0.392857 | 0.500000 | 0.440000 |
| LogisticRegression() | 0.392857 | 0.500000 | 0.440000 |
| MLPClassifier() | 0.392857 | 0.500000 | 0.440000 |

### Limit - Graphs Statistic - Binary

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| DecisionTreeClassifier() | 0.777778 | 0.777778 | 0.714286 |
| RandomForestClassifier() | 0.777778 | 0.777778 | 0.714286 |
| AdaBoostClassifier() | 0.777778 | 0.777778 | 0.714286 |
| ExtraTreesClassifier() | 0.777778 | 0.777778 | 0.714286 |
| XGBClassifier() | 0.666667 | 0.677778 | 0.641026 |

### Graphs Statistic - Multi-Labels - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| OneVsRestClassifier(GaussianNB()) | 0.465608 | 0.690476 | 0.505291 |
| MultiOutputClassifier(GaussianNB()) | 0.465608 | 0.690476 | 0.505291 |
| MultiOutputClassifier(XGBClassifier()) | 0.428571 | 0.559524 | 0.413105 |
| OneVsRestClassifier(XGBClassifier()) | 0.428571 | 0.559524 | 0.413105 |
| OneVsRestClassifier(AdaBoostClassifier()) | 0.312169 | 0.571429 | 0.359307 |

### Graphs Statistic - Binary - Gigahorse

#### Mint - Graphs Statistic - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| GaussianNB() | 0.642857 | 0.791667 | 0.590643 |
| XGBClassifier() | 0.642857 | 0.791667 | 0.590643 |
| SVC() | 0.625000 | 0.750000 | 0.533333 |
| DecisionTreeClassifier() | 0.611111 | 0.708333 | 0.475936 |
| AdaBoostClassifier() | 0.611111 | 0.708333 | 0.475936 |

#### Leak - Graphs Statistic - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| GaussianNB() | 0.742424 | 0.700 | 0.714286 |
| LogisticRegression() | 0.625000 | 0.575 | 0.575758 |
| XGBClassifier() | 0.625000 | 0.575 | 0.575758 |
| SVC() | 0.357143 | 0.500 | 0.416667 |
| MLPClassifier() | 0.357143 | 0.500 | 0.416667 |

#### Limit - Graphs Statistic - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| DecisionTreeClassifier() | 0.714286 | 0.714286 | 0.714286 |
| AdaBoostClassifier() | 0.714286 | 0.714286 | 0.714286 |
| ExtraTreesClassifier() | 0.714286 | 0.714286 | 0.714286 |
| RandomForestClassifier() | 0.645833 | 0.642857 | 0.641026 |
| KNeighborsClassifier() | 0.500000 | 0.500000 | 0.497436 |

## Opcode Entropy

### Opcode Entropy - Multi-Labels

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MPL 512 \* 256 \* 128 | 0.44 | 1.00 | 0.59 |
| MultiOutputClassifier(XGBClassifier()) | 0.533333 | 0.277778 | 0.348485 |
| OneVsRestClassifier(XGBClassifier()) | 0.533333 | 0.277778 | 0.348485 |
| MultiOutputClassifier(ExtraTreesClassifier()) | 0.261905 | 0.185185 | 0.205128 |
| MultiOutputClassifier(DecisionTreeClassifier()) | 0.261905 | 0.185185 | 0.205128 |
| OneVsRestClassifier(DecisionTreeClassifier()) | 0.261905 | 0.185185 | 0.205128 |

### Opcode Entropy - Binary

#### Mint - Opcode Entropy

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| XGBClassifier() | 0.633333 | 0.625000 | 0.625668 |
| LogisticRegression() | 0.285714 | 0.500000 | 0.363636 |
| LGBMClassifier() | 0.285714 | 0.500000 | 0.363636 |
| SVC() | 0.285714 | 0.500000 | 0.363636 |
| MLPClassifier() | 0.285714 | 0.500000 | 0.363636 |

#### Leak - Opcode Entropy

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| LogisticRegression() | 0.392857 | 0.5 | 0.441 |
| DecisionTreeClassifier() | 0.392857 | 0.5 | 0.442 |
| RandomForestClassifier() | 0.392857 | 0.5 | 0.443 |
| AdaBoostClassifier() | 0.392857 | 0.5 | 0.444 |
| ExtraTreesClassifier() | 0.392857 | 0.5 | 0.44 |

#### Limit - Opcode Entropy

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| XGBClassifier() | 0.727273 | 0.666667 | 0.5625001 |
| SVC() | 0.575000 | 0.566667 | 0.4974362 |
| KNeighborsClassifier() | 0.575000 | 0.566667 | 0.4974363 |
| LGBMClassifier() | 0.428571 | 0.422222 | 0.4166674 |
| SGDClassifier() | 0.321429 | 0.500000 | 0.391304 |

## Opcode Frequency

### Opcode Frequency - Multi-Labels

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| OneVsRestClassifier(KNeighborsClassifier()) | 0.866667 | 0.500000 | 0.599415 |
| MultiOutputClassifier(KNeighborsClassifier()) | 0.866667 | 0.500000 | 0.599415 |
| MPL 512 \* 256 \* 128 | 0.43 | 1.00 | 0.58 |
| OneVsRestClassifier(AdaBoostClassifier()) | 0.722222 | 0.370370 | 0.477778 |
| MultiOutputClassifier(GaussianNB()) | 0.421645 | 0.537037 | 0.470513 |
| OneVsRestClassifier(GaussianNB()) | 0.421645 | 0.537037 | 0.470513 |

#### Opcode Frequency - Multi-Labels -- gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| OneVsRestClassifier(KNeighborsClassifier()) | 0.515873 | 0.654762 | 0.515873 |
| MultiOutputClassifier(KNeighborsClassifier()) | 0.515873 | 0.654762 | 0.515873 |
| OneVsRestClassifier(SGDClassifier()) | 0.332418 | 0.952381 | 0.481481 |
| MultiOutputClassifier(GaussianNB()) | 0.422619 | 0.571429 | 0.462963 |
| OneVsRestClassifier(GaussianNB()) | 0.422619 | 0.571429 | 0.462963 |

### Opcode Frequency - Binary

#### Mint - Opcode Frequency

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| KNeighborsClassifier() | 0.863636 | 0.750000 | 0.754386 |
| DecisionTreeClassifier() | 0.833333 | 0.666667 | 0.650000 |
| LGBMClassifier() | 0.833333 | 0.666667 | 0.650000 |
| AdaBoostClassifier() | 0.833333 | 0.666667 | 0.650000 |
| SGDClassifier() | 0.633333 | 0.625000| 0.625668 |

#### Leak - Opcode Frequency

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| KNeighborsClassifier() | 0.923077 | 0.666667 | 0.708333 |
| AdaBoostClassifier() | 0.666667 | 0.621212 | 0.634783 |
| GaussianNB() | 0.488889 | 0.484848 | 0.475000 |
| RandomForestClassifier() | 0.392857 | 0.500000 | 0.440000 |
| SGDClassifier() | 0.392857 | 0.500000 | 0.440000 |

#### Limit - Opcode Frequency

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| LogisticRegression() | 0.666667 | 0.677778 | 0.641026 |
| DecisionTreeClassifier() | 0.666667 | 0.677778 | 0.641026 |
| MLPClassifier() | 0.600000 | 0.588889 | 0.590643 |
| RandomForestClassifier() | 0.571429 | 0.577778 | 0.562500 |
| LGBMClassifier() | 0.571429 | 0.577778 | 0.562500 |

## Opcode N Grams

### Opcode N Grams - Multi-Labels

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MultiOutputClassifier(XGBClassifier()) | 0.916667 | 0.574074 | 0.704762 |
| OneVsRestClassifier(XGBClassifier()) | 0.916667 | 0.574074 | 0.704762 |
| MPL 512 \* 256 \* 128 | 0.71 | 0.72 | 0.64 |
| MultiOutputClassifier(ExtraTreesClassifier()) | 0.822222 | 0.481481 | 0.579365 |
| MultiOutputClassifier(GaussianNB()) | 0.821429 | 0.462963 | 0.575000 |
| OneVsRestClassifier(GaussianNB()) | 0.821429 | 0.462963 | 0.575000 |

#### Opcode N Grams - Multi-Labels - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MPL 512 \* 256 \* 128 | 0.53 | 1.00 | 0.68 |
| MultiOutputClassifier(LogisticRegression()) | 0.611111 | 0.702381 | 0.585470 |
| OneVsRestClassifier(LogisticRegression()) | 0.611111 | 0.702381 | 0.585470 |
| OneVsRestClassifier(AdaBoostClassifier()) | 0.528571 | 0.690476 | 0.537037 |
| MultiOutputClassifier(AdaBoostClassifier()) | 0.528571 | 0.690476 | 0.537037 |
| MultiOutputClassifier(RandomForestClassifier()) | 0.349206 | 0.571429 | 0.404762 |

### Opcode N Grams - Binary

#### Mint - Opcode N Grams

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| DecisionTreeClassifier() | 0.708333 | 0.708333 | 0.708333 |
| XGBClassifier() | 0.725000 | 0.687500 | 0.688889 |
| GaussianNB() | 0.725000 | 0.687500 | 0.688889 |
| KNeighborsClassifier() | 0.633333 | 0.625000 | 0.625668 |
| SVC() | 0.550000 | 0.541667 | 0.533333 |

#### Leak - Opcode N Grams

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| DecisionTreeClassifier() | 0.708333 | 0.708333 | 0.708333 |
| XGBClassifier() | 0.725000 | 0.687500 | 0.688889 |
| GaussianNB() | 0.725000 | 0.687500 | 0.688889 |
| KNeighborsClassifier() | 0.633333 | 0.625000 | 0.625668 |
| SVC() | 0.550000 | 0.541667 | 0.533333 |

#### Limit - Opcode N Grams

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| XGBClassifier() | 0.777778 | 0.777778 | 0.714286 |
| LogisticRegression() | 0.714286 | 0.733333 | 0.708333 |
| AdaBoostClassifier() | 0.750000 | 0.722222 | 0.641026 |
| SGDClassifier() | 0.600000 | 0.588889 | 0.590643 |
| MLPClassifier() | 0.622222 | 0.622222 | 0.571429 |

#### Mint - Opcode N Grams - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MLPClassifier() | 0.700000 | 0.875000 | 0.714286 |
| SVC() | 0.700000 | 0.875000 | 0.714286 |
| LogisticRegression() | 0.666667 | 0.833333 | 0.650000 |
| KNeighborsClassifier() | 0.666667 | 0.833333 | 0.650000 |
| GaussianNB() | 0.642857 | 0.791667 | 0.590643 |

#### Leak - Opcode N Grams - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| AdaBoostClassifier() | 0.650000 | 0.650 | 0.650000 |
| LogisticRegression() | 0.625000 | 0.575 | 0.575758 |
| DecisionTreeClassifier() | 0.530303 | 0.525 | 0.523810 |
| SGDClassifier() | 0.475000 | 0.475 | 0.475000 |
| MLPClassifier() | 0.357143 | 0.500 | 0.416667 |

#### Limit - Opcode N Grams - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| LogisticRegression() | 0.937500 | 0.928571 | 0.928205 |
| ExtraTreesClassifier() | 0.791667 | 0.785714 | 0.784615 |
| XGBClassifier() | 0.714286 | 0.714286 | 0.714286 |
| RandomForestClassifier() | 0.714286 | 0.714286 | 0.714286 |
| AdaBoostClassifier() | 0.733333 | 0.714286 | 0.708333 |

## TF-IDF (Term Frequency-Inverse Document Frequency)

### Source Code

#### Source Code - Multi-Labels

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MultiOutputClassifier(DecisionTreeClassifier()) | 0.587302 | 0.750000 | 0.648485 |
| MultiOutputClassifier(SGDClassifier()) | 0.524242 | 0.833333 | 0.605820 |
| OneVsRestClassifier(AdaBoostClassifier()) | 0.611111 | 0.638889 | 0.600000 |
| MultiOutputClassifier(AdaBoostClassifier()) | 0.611111 | 0.638889 | 0.600000 |
| OneVsRestClassifier(MLPClassifier()) | 0.611111 | 0.527778 | 0.533333 |

#### Source Code - Binary

##### Mint - Source Code

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| DecisionTreeClassifier() | 1.000000 | 1.000000 | 1.000000 |
| XGBClassifier() | 0.875000 | 0.954545 | 0.904762 |
| LGBMClassifier() | 0.958333 | 0.833333 | 0.878261 |
| AdaBoostClassifier() | 0.958333 | 0.833333 | 0.878261 |
| GaussianNB() | 0.923077 | 0.666667 | 0.708333 |

##### Leak - Source Code

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MLPClassifier() | 0.961538 | 0.750000 | 0.813333 |
| DecisionTreeClassifier() | 0.621212 | 0.666667 | 0.634783 |
| AdaBoostClassifier() | 0.621212 | 0.666667 | 0.634783 |
| GaussianNB() | 0.544444 | 0.583333 | 0.523810 |
| SGDClassifier() | 0.544444 | 0.583333 | 0.523810 |

##### Limit - Source Code

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MLPClassifier() | 0.961538 | 0.750000 | 0.813333 |
| DecisionTreeClassifier() | 0.621212 | 0.666667 | 0.634783 |
| AdaBoostClassifier() | 0.621212 | 0.666667 | 0.634783 |
| GaussianNB() | 0.544444 | 0.583333 | 0.523810 |
| SGDClassifier() | 0.544444 | 0.583333 | 0.523810 |

### Opcode

#### Opcode - Multi-Labels

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| MultiOutputClassifier(AdaBoostClassifier()) | 0.916667 | 0.722222 | 0.774603 |
| OneVsRestClassifier(AdaBoostClassifier()) | 0.916667 | 0.722222 | 0.774603 |
| OneVsRestClassifier(ExtraTreesClassifier()) | 0.611111 | 0.444444 | 0.500000 |
| MultiOutputClassifier(XGBClassifier()) | 0.555556 | 0.500000 | 0.488889 |
| OneVsRestClassifier(XGBClassifier()) | 0.555556 | 0.500000 | 0.488889 |

#### Opcode - Multi-Labels - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| OneVsRestClassifier(GaussianNB()) | 0.477778 | 0.616667 | 0.537374 |
| MultiOutputClassifier(GaussianNB()) | 0.477778 | 0.616667 | 0.537374 |
| MultiOutputClassifier(DecisionTreeClassifier()) | 0.385185 | 0.583333 | 0.460317 |
| MultiOutputClassifier(AdaBoostClassifier()) | 0.533333 | 0.366667 | 0.422222 |
| OneVsRestClassifier(AdaBoostClassifier()) | 0.533333 | 0.366667 | 0.422222 |

#### Opcode - Binary

##### Mint - Opcode

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| AdaBoostClassifier() | 0.900000 | 0.833333 | 0.844444 |
| KNeighborsClassifier() | 0.788889 | 0.770833 | 0.775401 |
| RandomForestClassifier() | 0.863636 | 0.750000 | 0.754386 |
| DecisionTreeClassifier() | 0.863636 | 0.750000 | 0.754386 |
| ExtraTreesClassifier() | 0.863636 | 0.750000 | 0.754386 |

##### Leak - Opcode

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| AdaBoostClassifier() | 0.961538 | 0.750 | 0.813333 |
| LogisticRegression() | 0.428571 | 0.500 | 0.461538 |
| DecisionTreeClassifier() | 0.428571 | 0.500 | 0.461538 |
| RandomForestClassifier() | 0.428571 | 0.500 | 0.461538 |
| ExtraTreesClassifier() | 0.428571 | 0.500 | 0.461538 |

##### Limit - Opcode

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| AdaBoostClassifier() | 0.875000 | 0.875000 | 0.857143 |
| ExtraTreesClassifier() | 0.854167 | 0.854167 | 0.854167 |
| LGBMClassifier() | 0.785714 | 0.791667 | 0.784615 |
| RandomForestClassifier() | 0.785714 | 0.791667 | 0.784615 |
| XGBClassifier() | 0.729167 | 0.729167 | 0.714286 |

##### Mint - Opcode - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| GaussianNB() | 0.742424 | 0.700 | 0.714286 |
| RandomForestClassifier() | 0.884615 | 0.625 | 0.634783 |
| ExtraTreesClassifier() | 0.884615 | 0.625 | 0.634783 |
| XGBClassifier() | 0.884615 | 0.625 | 0.634783 |
| DecisionTreeClassifier() | 0.588889 | 0.600 | 0.590643 |

##### Leak - Opcode - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| LogisticRegression() | 0.428571 | 0.5 | 0.461538 |
| DecisionTreeClassifier() | 0.428571 | 0.5 | 0.461538 |
| RandomForestClassifier() | 0.428571 | 0.5 | 0.461538 |
| AdaBoostClassifier() | 0.428571 | 0.5 | 0.461538 |
| ExtraTreesClassifier() | 0.428571 | 0.5 | 0.461538 |

##### Limit - Opcode - Gigahorse

| Model | Precision | Recall | F1-score |
| - | - | - | - |
| ExtraTreesClassifier() | 0.770833 | 0.788889 | 0.775401 |
| DecisionTreeClassifier() | 0.714286 | 0.733333 | 0.708333 |
| XGBClassifier() | 0.714286 | 0.733333 | 0.708333 |
| RandomForestClassifier() | 0.714286 | 0.733333 | 0.708333 |
| LGBMClassifier() | 0.714286 | 0.733333 | 0.708333 |

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import GRU, Dense, Masking
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

class GRUBlocks:
    @staticmethod
    def scheduler(epoch, lr): return lr if epoch < 10 else lr * 0.9
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    lr_sched   = LearningRateScheduler(scheduler)

    @staticmethod
    def build_classifier(input_shape, units, lr, output):
        m = Sequential([Input(shape=input_shape), Masking(mask_value=0.0), GRU(units), Dense(output, activation="sigmoid")])
        m.compile(optimizer=Adam(learning_rate=lr), loss=BinaryCrossentropy())
        return m

    @staticmethod
    def build_autoencoder(input_shape, units, lr):
        m = Sequential([
            Input(shape=input_shape),
            Masking(mask_value=0.0),
            GRU(units, return_sequences=True),
            Dense(units // 2, activation="relu"),
            Dense(input_shape[1], activation="linear")
        ])
        m.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        return m

    @staticmethod
    def train(model, Xtr, Xval, ytr, yval, params):
        hist = model.fit(
            Xtr, ytr,
            validation_data=(Xval, yval),
            epochs=int(params.get("epochs", 20)),
            batch_size=int(params.get("batch_size", 64)),
            callbacks=[GRUBlocks.early_stop, GRUBlocks.reduce_lr, GRUBlocks.lr_sched],
            verbose=1
        )
        return model, hist

class SklearnFactory:
    @staticmethod
    def to_float(x):
        return x.astype(np.float32)

    @staticmethod
    def build_model_by_name(name, mode, param_source, is_trial=False, random_state=42):
        if is_trial:
            trial = param_source
            gF = lambda n, *a, **k: trial.suggest_float(n, *a, **k)
            gI = lambda n, *a, **k: trial.suggest_int(n, *a, **k)
            gC = lambda n, *a, **k: trial.suggest_categorical(n, *a, **k)
        else:
            params = param_source
            gF = lambda n, *a, **k: params[n]; gI = gF; gC = gF

        match name:
            case "LogisticRegression":
                base = LogisticRegression(C=gF("C", 0.01, 10.0), class_weight="balanced", solver="liblinear", random_state=random_state)
            case "RandomForest":
                base = RandomForestClassifier(n_estimators=gI("n_estimators", 50, 200), max_depth=gI("max_depth", 3, 20),
                                              class_weight="balanced", random_state=random_state)
            case "BaggingClassifier":
                base = BaggingClassifier(n_estimators=gI("n_estimators", 10, 100), random_state=random_state)
            case "XGBoost":
                base = XGBClassifier(n_estimators=gI("n_estimators", 50, 200), max_depth=gI("max_depth", 3, 20),
                                     learning_rate=gF("learning_rate", 0.01, 0.3),
                                     use_label_encoder=False, eval_metric="logloss", verbosity=0, random_state=random_state)
            case "LightGBM":
                base = LGBMClassifier(n_estimators=gI("n_estimators", 50, 200), max_depth=gI("max_depth", 3, 20),
                                      learning_rate=gF("learning_rate", 0.01, 0.3), random_state=random_state, verbose=-1)
            case "MLP":
                hidden = {"32":(32,), "64":(64,), "128":(128,), "256":(256,),
                          "32_64":(32,64), "64_32":(64,32), "64_64":(64,64),
                          "64_128":(64,128), "128_128":(128,128), "32_64_128":(32,64,128)}
                key = gC("hidden_layer_sizes", list(hidden.keys()))
                base = MLPClassifier(hidden_layer_sizes=hidden[key], learning_rate_init=gF("learning_rate_init", 1e-4, 1e-1),
                                     max_iter=300, random_state=random_state)
            case _:
                raise ValueError(f"Unknown model: {name}")

        base = MultiOutputClassifier(base)

        match mode:
            case "general":
                return Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler(with_mean=False)), ("clf", base)])
            case "sol":
                return Pipeline([("tfidf", TfidfVectorizer(lowercase=True, analyzer="word", token_pattern=r"\b\w+\b",
                                                           max_features=gI("n_max_features", 10, 20000), min_df=gI("n_min_df", 1, 5))),
                                 ("clf", base)])
            case "opcode":
                ngram = {"1":(1,1), "2":(1,2), "3":(1,3)}
                key = gC("ngram_range", list(ngram.keys()))
                return Pipeline([("count", CountVectorizer(analyzer="word", ngram_range=ngram[key],
                                                           max_features=gI("n_max_features", 10, 20000), min_df=gI("n_min_df", 1, 5))),
                                 ("to_float", FunctionTransformer(SklearnFactory.to_float, accept_sparse=True)),
                                 ("clf", base)])
            case _:
                raise ValueError(f"Unknown mode: {mode}")

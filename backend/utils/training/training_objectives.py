import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import GRU, Dense, Masking, LSTM, Bidirectional
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

class TSBlocks:
    @staticmethod
    def scheduler(epoch, lr): return lr if epoch < 10 else lr * 0.9
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    lr_sched   = LearningRateScheduler(scheduler)

    @staticmethod
    def build_classifier(input_shape, units, lr, output):
        m = Sequential([
            Input(shape=input_shape),
            Masking(mask_value=0.0),
            GRU(units),  # keep GRU here, or switch to Bidirectional(LSTM(units))
            Dense(output, activation="sigmoid")
        ])
        m.compile(optimizer=Adam(learning_rate=lr), loss=BinaryCrossentropy())
        return m

    @staticmethod
    def build_autoencoder(input_shape, units, lr):
        timesteps, features = input_shape
        m = Sequential([
            Input(shape=input_shape),
            Masking(mask_value=0.0),
            Bidirectional(LSTM(units, return_sequences=True)),  # ✅ return full sequence
            Dense(units // 2, activation="relu"),               # applied per time step
            Dense(features, activation="linear")                # per time step → (T, F)
        ])
        m.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        return m

    @staticmethod
    def train(model, Xtr, Xval, ytr, yval, params):
        cbs = [TSBlocks.early_stop, TSBlocks.reduce_lr, TSBlocks.lr_sched]
        if "pruning_cb" in params and params["pruning_cb"] is not None:
            cbs.append(params["pruning_cb"])
        hist = model.fit(
            Xtr, ytr,
            validation_data=(Xval, yval),
            epochs=int(params.get("epochs", 20)),
            batch_size=int(params.get("batch_size", 64)),
            callbacks=cbs,
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
                penalty = gC('penalty', ['l2', 'l1', 'elasticnet'])
                solver = "liblinear" if penalty in ("l1", "l2") else "saga"
                base =  LogisticRegression(
                    penalty=penalty,
                    class_weight='balanced',
                    tol=gF('tol', 1e-10, 1e-2),
                    C=gF("C", 1e-4, 10.0, log=True),
                    solver=solver,
                    l1_ratio=(gF("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else None),
                    random_state=random_state,
                    max_iter=2000,
                    verbose=0
                )
            case "RandomForest":
                base = RandomForestClassifier(
                    n_estimators=gI("n_estimators", 50, 1000, step=50),
                    criterion=gC('criterion', ['gini', 'entropy', 'log_loss']),
                    max_depth=gI("max_depth", 5, 30, step=5),
                    class_weight='balanced',
                    random_state=random_state,
                    verbose=0
                )
            case "BaggingClassifier":
                base = BaggingClassifier(n_estimators=gI("n_estimators", 50, 1000, step=50),
                                         random_state=random_state)
            case "XGBoost":
                base = XGBClassifier(
                    n_estimators=gI("n_estimators", 50, 1000, step=50),
                    max_depth=gI("max_depth", 5, 30, step=5),
                    learning_rate=gF("learning_rate", 0.01, 1.0),
                    subsample=gF("subsample", 0.5, 1.0, step=0.1),
                    colsample_bytree=gF("colsample_bytree", 0.5, 1.0, step=0.1),
                    eval_metric="logloss",
                    base_score=0.5,
                    objective="binary:logistic",
                    random_state=random_state
                )
            case "LightGBM":
                base = LGBMClassifier(
                    n_estimators=gI("n_estimators", 50, 1000, step=50),
                    max_depth=gI("max_depth", 5, 30, step=5),
                    learning_rate=gF("learning_rate", 0.01, 1.0),
                    subsample=gF("subsample", 0.5, 1.0, step=0.1),
                    colsample_bytree=gF("colsample_bytree", 0.5, 1.0, step=0.1),
                    objective="binary",
                    random_state=random_state,
                    verbose=-1
                )
            case "MLP":
                hidden = {"32":(32,), "64":(64,), "128":(128,), "256":(256,),
                          "32_64":(32,64), "64_32":(64,32), "64_64":(64,64),
                          "64_128":(64,128), "128_128":(128,128), "32_64_128":(32,64,128)}
                key = gC("hidden_layer_sizes", list(hidden.keys()))
                base = MLPClassifier(
                    hidden_layer_sizes=hidden[key],
                    activation=gC("activation", ["relu", "tanh", 'identity', 'logistic']),
                    learning_rate_init=gF("learning_rate_init", 1e-10, 1e-1),
                    max_iter=500,
                    random_state=random_state,
                    verbose=0
                )
            case _:
                raise ValueError(f"Unknown model: {name}")

        base = MultiOutputClassifier(base)

        match mode:
            case "general":
                return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                                 ("scaler", StandardScaler(with_mean=False)),
                                 ("clf", base)]).set_output(transform="pandas")
            case "sol":
                return Pipeline([("tfidf", TfidfVectorizer(lowercase=True, analyzer="word", token_pattern=r"\b\w+\b",
                                                           max_features=gI("n_max_features", 10, 20000),
                                                           min_df=gI("n_min_df", 1, 5))),
                                 ("clf", base)])
            case "opcode":
                ngram = {"1":(1,1), "2":(1,2), "3":(1,3)}
                key = gC("ngram_range", list(ngram.keys()))
                return Pipeline([("count", CountVectorizer(analyzer="word", ngram_range=ngram[key],
                                                           max_features=gI("n_max_features", 10, 20000),
                                                           min_df=gI("n_min_df", 1, 5))),
                                 ("to_float", FunctionTransformer(SklearnFactory.to_float, accept_sparse=True)),
                                 ("clf", base)])
            case _:
                raise ValueError(f"Unknown mode: {mode}")

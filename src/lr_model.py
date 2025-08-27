import numpy as np
import pandas as pd
import logging

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.compose import TransformedTargetRegressor

from prepare_data import read_data, split_num_cat, FEATURES

RANDOM_STATE = 42

logging.basicConfig(
    filename='lr_log.log',
    level=logging.INFO,
    format='%(message)s'
)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_preprocessor(train_df: pd.DataFrame):
    num_cols, cat_cols = split_num_cat(train_df)

    num_pipe = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.05))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def make_model(base_estimator):
    # обучаем на log1p, предсказываем в долларах
    return TransformedTargetRegressor(
        regressor=base_estimator,
        func=np.log1p,
        inverse_func=np.expm1
    )

def evaluate_cv(pipeline, X, y, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    scores = cross_val_score(pipeline, X, y, scoring=rmse_scorer, cv=cv, n_jobs=-1)
    return -scores.mean()

def main():
    train_df = pd.read_csv("../data/prep_train.csv")

    X = train_df[FEATURES]
    y = train_df["SalePrice"]

    preprocessor = get_preprocessor(train_df)

    # base lr
    lr = make_model(LinearRegression(n_jobs=-1))
    pipe_lr = Pipeline(steps=[("pre", preprocessor), ("model", lr)])
    rmse_lr = evaluate_cv(pipe_lr, X, y)
    logging.info(f"Baseline CV RMSE: {rmse_lr:,.0f} $")

if __name__ == "__main__":
    main()

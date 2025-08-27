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
    pipe_base = Pipeline(steps=[("pre", preprocessor), ("model", lr)])
    rmse_lr = evaluate_cv(pipe_base, X, y)
    logging.info(f"Baseline CV RMSE: {rmse_lr:,.0f} $")

    # ridge
    ridge = make_model(Ridge(random_state=RANDOM_STATE, max_iter=5000))
    pipe_ridge = Pipeline(steps=[("pre", preprocessor), ("model", ridge)])
    param_grid_ridge = {
        "model__regressor__alpha": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    }
    gs_ridge = GridSearchCV(
        estimator=pipe_ridge,
        param_grid=param_grid_ridge,
        scoring=make_scorer(rmse, greater_is_better=False),
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    gs_ridge.fit(X, y)
    best_ridge_rmse = -gs_ridge.best_score_
    best_ridge_alpha = gs_ridge.best_params_["model__regressor__alpha"]
    logging.info(f"Ridge: Best alpha={best_ridge_alpha}, CV RMSE: {best_ridge_rmse:,.0f} $")

    # lasso
    lasso = make_model(Lasso(random_state=RANDOM_STATE, max_iter=10000))
    pipe_lasso = Pipeline(steps=[("pre", preprocessor), ("model", lasso)])
    param_grid_lasso = {
        "model__regressor__alpha": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
    }
    gs_lasso = GridSearchCV(
        estimator=pipe_lasso,
        param_grid=param_grid_lasso,
        scoring=make_scorer(rmse, greater_is_better=False),
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    gs_lasso.fit(X, y)
    best_lasso_rmse = -gs_lasso.best_score_
    best_lasso_alpha = gs_lasso.best_params_["model__regressor__alpha"]
    logging.info(f"Lasso: Best alpha={best_lasso_alpha}, CV RMSE: {best_lasso_rmse:,.0f} $")

    # best
    model_name, best_estimator, best_cv_rmse = min(
        [
            ("base", pipe_base.fit(X, y), rmse_lr),
            ("ridge", gs_ridge.best_estimator_, best_ridge_rmse),
            ("lasso", gs_lasso.best_estimator_, best_lasso_rmse),
        ],
        key=lambda t: t[2]
    )
    logging.info(f"Chosen model: {model_name} with CV RMSE {best_cv_rmse:,.0f} $")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import logging

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import make_scorer

import utils
from prepare_data import FEATURES

RANDOM_STATE = 42

logging.basicConfig(
    filename='../logs/models_score.log',
    level=logging.INFO,
    format='%(message)s'
)

def train_lr(train_df: pd.DataFrame, test_df: pd.DataFrame):

    cols = [c for c in FEATURES if c in train_df.columns and c in test_df.columns]
    X = train_df[FEATURES]
    y = train_df["SalePrice"]
    X_test = test_df[cols]

    preprocessor = utils.get_preprocessor(train_df)

    # base lr
    lr_pipe = utils.make_pipeline(utils.make_model(LinearRegression()), preprocessor)
    rmse_lr = utils.evaluate_cv(lr_pipe, X, y)
    logging.info(f"Baseline CV RMSE: {rmse_lr:,.0f} $")

    # ridge
    ridge_best_pipe, ridge_best_rmse, ridge_best_params = utils.tune_model_grid(
        Ridge(random_state=RANDOM_STATE, max_iter=5000),
        preprocessor,
        X, y,
        param_grid={"model__regressor__alpha": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]},
        scoring=make_scorer(utils.rmse, greater_is_better=False)
    )
    logging.info(f"Ridge: Best alpha={ridge_best_params['model__regressor__alpha']}, CV RMSE: {ridge_best_rmse:,.0f} $")

    # lasso
    lasso_best_pipe, lasso_best_rmse, lasso_best_params = utils.tune_model_grid(
        Lasso(random_state=RANDOM_STATE, max_iter=10000),
        preprocessor,
        X, y,
        param_grid={"model__regressor__alpha": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]},
        scoring=make_scorer(utils.rmse, greater_is_better=False)
    )
    logging.info(f"Lasso: Best alpha={lasso_best_params['model__regressor__alpha']}, CV RMSE: {lasso_best_rmse:,.0f} $")

    # best
    best_model_name, best_estimator, best_cv_rmse = min(
        [
            ("base", lr_pipe, rmse_lr),
            ("ridge", ridge_best_pipe, ridge_best_rmse),
            ("lasso", lasso_best_pipe, lasso_best_rmse),
        ],
        key=lambda t: t[2]
    )

    logging.info(f"Chosen model: {best_model_name} with CV RMSE {best_cv_rmse:,.0f} $")

    best_estimator.fit(X, y)
    preds = best_estimator.predict(X_test)

    return preds, best_model_name
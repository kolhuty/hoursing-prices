import numpy as np
import pandas as pd
import logging

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from src.utils import RANDOM_STATE

import utils
from prepare_data import FEATURES

logging.basicConfig(
    filename='../logs/models_score.log',
    level=logging.INFO,
    format='%(message)s'
)

def find_the_best_model(train_df: pd.DataFrame) -> tuple:

    X = train_df[FEATURES]
    y = train_df["SalePrice"]

    preprocessor = utils.get_preprocessor(train_df, normalizing=False)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE)
    }

    param_grids = {
        "Ridge": {
            "model__regressor__alpha": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        },
        "Lasso": {
            "model__regressor__alpha": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
        },
        "DecisionTree": {
            "model__regressor__max_depth": [5, 7, 9, 10, None],
            "model__regressor__min_samples_leaf": [1, 5, 8]
        },
        "RandomForest": {
            "model__regressor__n_estimators": [100, 300],
            "model__regressor__max_depth": [10, 20, None],
            "model__regressor__min_samples_leaf": [1, 2, 5]
        },
        "GradientBoosting": {
            "model__regressor__n_estimators": [100, 300],
            "model__regressor__learning_rate": [0.05, 0.1],
            "model__regressor__max_depth": [3, 5]
        }
    }

    results = {}

    for name, model in models.items():
        if name == "LinearRegression":
            pipe = utils.make_pipeline(utils.make_model(model), preprocessor)
            rmse = utils.evaluate_cv(pipe, X, y)
            results[name] = [pipe, rmse, None]
            continue

        best_pipe, best_rmse, best_params = utils.tune_model_grid(
            model,
            preprocessor,
            X, y,
            param_grids[name],
            scoring=make_scorer(utils.rmse, greater_is_better=False)
        )
        results[name] = [best_pipe, best_rmse, best_params]

    for name, (best_pipe, best_rmse, best_params) in results.items():
        if best_params:
            logging.info(
                f"{name}: Best params={best_params}\n"
                f"CV RMSE: {best_rmse:,.0f}$"
            )
        else:
            logging.info(f"{name}: CV RMSE: {best_rmse:,.0f}$")

    best_model_name, best_estimator, best_cv_rmse = min(
        [(name, res[0], res[1]) for name, res in results.items()],
        key=lambda t: t[2]
    )

    logging.info(f"Chosen model: {best_model_name} with CV RMSE {best_cv_rmse:,.0f}$")

    return best_model_name, best_estimator, best_cv_rmse
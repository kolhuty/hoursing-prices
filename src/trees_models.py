import numpy as np
import pandas as pd
import logging

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer

import utils
from prepare_data import FEATURES


RANDOM_STATE = 42

logging.basicConfig(
    filename='../logs/models_score.log',
    level=logging.INFO,
    format='%(message)s'
)

def train_trees(train_df: pd.DataFrame, test_df: pd.DataFrame):

    cols = [c for c in FEATURES if c in train_df.columns and c in test_df.columns]
    X = train_df[FEATURES]
    y = train_df["SalePrice"]
    X_test = test_df[cols]

    preprocessor = utils.get_preprocessor(train_df, normalizing=False)

    models = {
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    }

    param_grids = {
        "DecisionTree": {
            'model__regressor__max_depth': [5, 7, 9, 10, None],
            'model__regressor__min_samples_leaf': [1, 5, 8]
        },
        "RandomForest": {
            'model__regressor__n_estimators': [100, 300],
            'model__regressor__max_depth': [10, 20, None],
            'model__regressor__min_samples_leaf': [1, 2, 5]
        },
        "GradientBoosting": {
            'model__regressor__n_estimators': [100, 300],
            'model__regressor__learning_rate': [0.05, 0.1],
            'model__regressor__max_depth': [3, 5]
        },
        "XGBoost": {
            'model__regressor__n_estimators': [300, 500],
            'model__regressor__learning_rate': [0.05, 0.1],
            'model__regressor__max_depth': [3, 5, 7]
        }
    }

    grid_searches = {}

    for name in models.keys():
        best_pipe, best_rmse, best_params = utils.tune_model_grid(
                models[name],
                preprocessor,
                X, y,
                param_grids[name],
                scoring=make_scorer(utils.rmse, greater_is_better=False)
        )
        grid_searches[name] = [best_pipe, best_rmse, best_params]

    for name, gs in grid_searches.items():
        logging.info(f"{name}: Best params={grid_searches[name][2]}\n"
                     f"CV RMSE: {grid_searches[name][1]:,.0f} $")

    # best
    best_model_name, best_estimator, best_cv_rmse = min([
        (name, grid_searches[name][0], grid_searches[name][1])
        for name in grid_searches.keys()],
        key=lambda t: t[2]
    )

    logging.info(f"Chosen model: {best_model_name} with CV RMSE {best_cv_rmse:,.0f} $")

    best_estimator.fit(X, y)
    preds = best_estimator.predict(X_test)

    return preds, best_model_name

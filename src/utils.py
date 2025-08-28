import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, make_scorer

from prepare_data import split_num_cat

RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_preprocessor(train_df: pd.DataFrame, normalizing=True):
    num_cols, cat_cols = split_num_cat(train_df)

    if normalizing:
        num_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ("scaler", StandardScaler())
        ])
    else:
        num_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median', add_indicator=True))
        ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing', add_indicator=True)),
        ('ohe', OneHotEncoder(handle_unknown='ignore', min_frequency=0.05))
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

def make_pipeline(model, preprocessor):
    return Pipeline(steps=[("pre", preprocessor), ("model", model)])

def evaluate_model_cv(pipeline, X, y, cv=5, scoring=None):
    return evaluate_cv(pipeline, X, y) if scoring is None else -cross_val_score(
        pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1
    ).mean()

def tune_model_grid(model, preprocessor, X, y, param_grid, scoring, cv=5):
    pipe = make_pipeline(make_model(model), preprocessor)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    gs.fit(X, y)
    best_score = -gs.best_score_
    return gs.best_estimator_, best_score, gs.best_params_
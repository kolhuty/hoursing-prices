import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, make_scorer

from prepare_data import read_data, split_num_cat, FEATURES


RANDOM_STATE = 42

logging.basicConfig(
    filename='models_score.log',
    level=logging.INFO,
    format='%(message)s'
)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_preprocessor(train_df: pd.DataFrame):
    num_cols, cat_cols = split_num_cat(train_df)

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

def main():
    train_df, test_df = read_data("../data/train.csv", "../data/test.csv")
    cols = [c for c in FEATURES if c in train_df.columns and c in test_df.columns]
    X = train_df[FEATURES]
    y = train_df["SalePrice"]
    X_test = test_df[cols]

    preprocessor = get_preprocessor(train_df)

    dec_tree = make_model(DecisionTreeRegressor(random_state=RANDOM_STATE))
    pipe_dec_tree = Pipeline(steps=[("pre", preprocessor), ("model", dec_tree)])

    param_grid_dec_tree = {
        'model__regressor__max_depth': [5,7,9,12, None],
        'model__regressor__min_samples_leaf': [1,2,4,8,10]
    }

    gs_dec_tree = GridSearchCV(
        estimator=pipe_dec_tree,
        param_grid=param_grid_dec_tree,
        scoring=make_scorer(rmse, greater_is_better=False),
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    gs_dec_tree.fit(X, y)
    best_dec_tree_rmse = -gs_dec_tree.best_score_
    best_dec_tree_max_depth = gs_dec_tree.best_params_["model__regressor__max_depth"]
    best_dec_tree_min_samples_leaf = gs_dec_tree.best_params_["model__regressor__min_samples_leaf"]
    logging.info(f"Decision tree: Best decision tree max depth={best_dec_tree_max_depth}\nBest decision tree min samples leaf={best_dec_tree_min_samples_leaf}\nCV RMSE: {best_dec_tree_rmse:,.0f} $")

    dec_tree_model = gs_dec_tree.best_estimator_
    dec_tree_model.fit(X, y)
    preds = dec_tree_model.predict(X_test)

    # save
    submit = pd.DataFrame({"Id": test_df["Id"], "SalePrice": preds})
    out_path = f"../submissions/submission_dec_tree.csv"
    submit.to_csv(out_path, index=False)
    logging.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
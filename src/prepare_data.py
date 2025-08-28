import numpy as np
import pandas as pd

FEATURES = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']

def split_num_cat(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_force = ["MSSubClass", "MoSold", "YrSold"]
    cat_cols = [c for c in FEATURES if df[c].dtype == "object"]
    for c in cat_force:
        if c in FEATURES and c not in cat_cols:
            cat_cols.append(c)
    num_cols = [c for c in FEATURES if c not in cat_cols]
    return num_cols, cat_cols

def read_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def save_submit(Id: pd.Series, preds: np.ndarray, model_name: str):
    submit = pd.DataFrame({"Id": Id, "SalePrice": preds})
    out_path = f"../submissions/submission_{model_name}.csv"
    submit.to_csv(out_path, index=False)
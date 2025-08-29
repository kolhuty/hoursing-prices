import pandas as pd
from prepare_data import read_data
from model_selection import find_the_best_model
from prepare_data import FEATURES,save_submit

train_df, test_df = read_data("../data/train.csv", "../data/test.csv")

name, model, rmse = find_the_best_model(train_df)

X = train_df[FEATURES]
y = train_df["SalePrice"]
model.fit(X, y)
preds = model.predict(test_df)
save_submit(test_df["Id"], preds, name)
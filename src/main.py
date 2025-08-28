import pandas as pd

from src.prepare_data import read_data, save_submit
from src.lr_model import train_lr
from src.dec_tree_model import train_dec_tree


train_df, test_df = read_data("../data/train.csv", "../data/test.csv")
for train_func, model_name in [(train_lr, "lr"), (train_dec_tree, "dec_tree")]:
    preds, best_model_name = train_func(train_df, test_df)
    save_submit(test_df["Id"], preds, best_model_name)

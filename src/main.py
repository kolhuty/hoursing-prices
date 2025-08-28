from prepare_data import read_data, save_submit
from lr_models import train_lr
from trees_models import train_trees


train_df, test_df = read_data("../data/train.csv", "../data/test.csv")
for train_func, model_type in [(train_lr, "lr"), (train_trees, "tree")]:
    preds, best_model_name = train_func(train_df, test_df)
    save_submit(test_df["Id"], preds, best_model_name)

import gc
import pandas as pd
import warnings

import utils
import load_train
import load_hist
import load_new
import add_feature
import train

warnings.filterwarnings('ignore')


def main(debug=False):
    num_rows = 100000 if debug else None
    with utils.timer("train & test"):
        df = load_train.train_test(num_rows)
    with utils.timer("historical transactions"):
        df = pd.merge(df, load_hist.historical_transactions(num_rows), on='card_id', how='outer')
    with utils.timer("new merchants"):
        df = pd.merge(df, load_new.new_merchant_transactions(num_rows), on='card_id', how='outer')
    with utils.timer("additional features"):
        df = add_feature.additional_features(df)
    with utils.timer("split train & test"):
        train_df = df[df['target'].notnull()]
        test_df = df[df['target'].isnull()]
        target = df[df['target'].notnull()]['target']
        del df
        gc.collect()
    with utils.timer("Run LightGBM with kfold"):
        train.kfold_lightgbm(train_df, test_df, target, num_folds=11, stratified=False, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission.csv"
    with utils.timer("Full model run"):
        main(debug=True)

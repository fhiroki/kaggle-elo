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


def main(debug=False, is_load=True):
    num_rows = 100000 if debug else None
    if is_load:
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

            train_df.to_csv('../input/pre_process/train.csv')
            test_df.to_csv('../input/pre_process/test.csv')

            del df
            gc.collect()
    else:
        train_df = pd.read_csv('../input/pre_process/train.csv')
        test_df = pd.read_csv('../input/pre_process/test.csv')
        target = train_df['target']

    with utils.timer("Run LightGBM with kfold"):
        train.kfold_lightgbm(train_df, test_df, target, num_folds=11, stratified=False, debug=debug)


if __name__ == "__main__":
    with utils.timer("Full model run"):
        main(debug=False, is_load=False)

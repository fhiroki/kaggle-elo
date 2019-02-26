import datetime
import numpy as np
import pandas as pd

import utils


# preprocessing new_merchant_transactions
def new_merchant_transactions(num_rows=None):
    # load csv
    new_merchant_df = pd.read_csv('../input/new_merchant_transactions.csv', nrows=num_rows)

    # fillna
    new_merchant_df['category_2'].fillna(1.0, inplace=True)
    new_merchant_df['category_3'].fillna('A', inplace=True)
    new_merchant_df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
    new_merchant_df['installments'].replace(-1, np.nan, inplace=True)
    new_merchant_df['installments'].replace(999, np.nan, inplace=True)

    # trim
    # new_merchant_df['purchase_amount'] = new_merchant_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    new_merchant_df['authorized_flag'] = new_merchant_df['authorized_flag'].map({'Y': 1, 'N': -1}).astype(int)
    new_merchant_df['category_1'] = new_merchant_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    new_merchant_df['category_3'] = new_merchant_df['category_3'].map({'A': 0, 'B': 1, 'C': 2}).astype(int)

    # datetime features
    new_merchant_df['purchase_date'] = pd.to_datetime(new_merchant_df['purchase_date'])
    new_merchant_df['month'] = new_merchant_df['purchase_date'].dt.month
    new_merchant_df['day'] = new_merchant_df['purchase_date'].dt.day
    new_merchant_df['hour'] = new_merchant_df['purchase_date'].dt.hour
    new_merchant_df['weekofyear'] = new_merchant_df['purchase_date'].dt.weekofyear
    new_merchant_df['weekday'] = new_merchant_df['purchase_date'].dt.weekday
    new_merchant_df['weekend'] = (new_merchant_df['purchase_date'].dt.weekday >= 5).astype(int)

    # additional features
    new_merchant_df['price'] = new_merchant_df['purchase_amount'] / new_merchant_df['installments']

    # new_merchant_df['Good_Friday_2017'] = (pd.to_datetime('2017-04-14') - new_merchant_df['purchase_date'])\
    #     .dt.days.apply(lambda x: x if x > 0 and x < 10 else 0)
    # Tiradentes: April 21 2017
    # new_merchant_df['Tiradentes_Day_2017'] = (pd.to_datetime('2017-04-21') - new_merchant_df['purchase_date'])\
    #     .dt.days.apply(lambda x: x if x > 0 and x < 10 else 0)
    # Mothers Day: May 14 2017
    new_merchant_df['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04') - new_merchant_df['purchase_date'])\
        .dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Valentine's Day : June 12 2017
    new_merchant_df['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12') - new_merchant_df['purchase_date'])\
        .dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Fathers Day: August 13 2017
    new_merchant_df['Fathers_Day_2017'] = (pd.to_datetime('2017-08-13') - new_merchant_df['purchase_date'])\
        .dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Independence Day: September 7 2017
    # new_merchant_df['Independence_Day_2017'] = (pd.to_datetime('2017-09-07') - new_merchant_df['purchase_date'])\
    #     .dt.days.apply(lambda x: x if x > 0 and x < 50 else 0)
    # Childrens Day: October 12 2017
    new_merchant_df['Children_Day_2017'] = (pd.to_datetime('2017-10-12') - new_merchant_df['purchase_date'])\
        .dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Black Friday : November 24 2017
    new_merchant_df['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - new_merchant_df['purchase_date'])\
        .dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Christmas : December 25 2017
    new_merchant_df['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - new_merchant_df['purchase_date'])\
        .dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    # New Year's Day : January 01 2018
    new_merchant_df['New_Years_Day_2018'] = (pd.to_datetime('2018-01-01') - new_merchant_df['purchase_date'])\
        .dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Good Friday: March 30 2018
    # new_merchant_df['Good_Friday_2018'] = (pd.to_datetime('2018-03-30') - new_merchant_df['purchase_date'])\
    #     .dt.days.apply(lambda x: x if x > 0 and x < 10 else 0)
    # Tiradentes: April 21 2018
    # new_merchant_df['Tiradentes_Day_2018'] = (pd.to_datetime('2018-04-21') - new_merchant_df['purchase_date'])\
    #     .dt.days.apply(lambda x: x if x > 0 and x < 10 else 0)
    # Mothers Day: May 13 2018
    new_merchant_df['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - new_merchant_df['purchase_date'])\
        .dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    new_merchant_df['month_diff'] = ((datetime.datetime.today() - new_merchant_df['purchase_date']).dt.days)//30
    new_merchant_df['month_diff'] += new_merchant_df['month_lag']

    # additional features
    new_merchant_df['duration'] = new_merchant_df['purchase_amount']*new_merchant_df['month_diff']
    new_merchant_df['amount_month_ratio'] = new_merchant_df['purchase_amount']/new_merchant_df['month_diff']

    # reduce memory usage
    new_merchant_df = utils.reduce_mem_usage(new_merchant_df)

    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'weekend', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var', 'skew']
    aggs['installments'] = ['sum', 'max', 'mean', 'var', 'skew']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['month_diff'] = ['mean', 'var', 'skew']
    aggs['authorized_flag'] = ['mean', 'sum', 'min', 'max', 'var']
    # aggs['weekend'] = ['mean']
    # aggs['month'] = ['mean', 'min', 'max']
    # aggs['weekday'] = ['mean', 'min', 'max']
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size', 'count']
    aggs['price'] = ['mean', 'max', 'min', 'var']

    # aggs['Good_Friday_2017'] = ['mean']
    # aggs['Tiradentes_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['Fathers_Day_2017'] = ['mean']
    # aggs['Independence_Day_2017'] = ['mean']
    aggs['Children_Day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['New_Years_Day_2018'] = ['mean']
    # aggs['Good_Friday_2018'] = ['mean']
    # aggs['Tiradentes_Day_2018'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']

    aggs['duration'] = ['mean', 'min', 'max', 'var', 'skew']
    aggs['amount_month_ratio'] = ['mean', 'min', 'max', 'var', 'skew']

    for col in ['category_2', 'category_3']:
        new_merchant_df[col + '_mean'] = new_merchant_df.groupby([col])['purchase_amount'].transform('mean')
        new_merchant_df[col + '_min'] = new_merchant_df.groupby([col])['purchase_amount'].transform('min')
        new_merchant_df[col + '_max'] = new_merchant_df.groupby([col])['purchase_amount'].transform('max')
        new_merchant_df[col + '_sum'] = new_merchant_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    new_merchant_df = new_merchant_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    new_merchant_df.columns = pd.Index([e[0] + "_" + e[1] for e in new_merchant_df.columns.tolist()])
    new_merchant_df.columns = ['new_' + c for c in new_merchant_df.columns]

    new_merchant_df['new_purchase_date_diff'] = \
        (new_merchant_df['new_purchase_date_max'] - new_merchant_df['new_purchase_date_min']).dt.days
    new_merchant_df['new_purchase_date_average'] = \
        new_merchant_df['new_purchase_date_diff']/new_merchant_df['new_card_id_size']
    new_merchant_df['new_purchase_date_uptonow'] = \
        (datetime.datetime.today()-new_merchant_df['new_purchase_date_max']).dt.days
    new_merchant_df['new_purchase_date_uptomin'] = \
        (datetime.datetime.today()-new_merchant_df['new_purchase_date_min']).dt.days

    # reduce memory usage
    new_merchant_df = utils.reduce_mem_usage(new_merchant_df)

    return new_merchant_df

import datetime
import numpy as np
import pandas as pd

import utils


# preprocessing historical transactions
def historical_transactions(num_rows=None):
    # load csv
    hist_df = pd.read_csv('../input/historical_transactions.csv', nrows=num_rows)

    # fillna
    hist_df['category_2'].fillna(1.0, inplace=True)
    hist_df['category_3'].fillna('A', inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
    hist_df['installments'].replace(-1, np.nan, inplace=True)
    hist_df['installments'].replace(999, np.nan, inplace=True)

    # trim
    # hist_df['purchase_amount'] = hist_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': -1}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A': 0, 'B': 1, 'C': 2})

    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >= 5).astype(int)

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']

    # Good Friday: April 14 2017
    # hist_df['Good_Friday_2017'] = (pd.to_datetime('2017-04-14') - hist_df['purchase_date']).dt.days.apply(
    #     lambda x: x if x > 0 and x < 10 else 0)
    # Tiradentes: April 21 2017
    # hist_df['Tiradentes_Day_2017'] = (pd.to_datetime('2017-04-21') - hist_df['purchase_date']).dt.days.apply(
    #     lambda x: x if x > 0 and x < 10 else 0)
    # Mothers Day: May 14 2017
    hist_df['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Valentine's Day : June 12 2017
    hist_df['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Fathers Day: August 13 2017
    hist_df['Fathers_Day_2017'] = (pd.to_datetime('2017-08-13') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Independence Day: September 7 2017
    # hist_df['Independence_Day_2017'] = (pd.to_datetime('2017-09-07') - hist_df['purchase_date']).dt.days.apply(
    #     lambda x: x if x > 0 and x < 50 else 0)
    # Childrens Day: October 12 2017
    hist_df['Children_Day_2017'] = (pd.to_datetime('2017-10-12') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Black Friday : November 24 2017
    hist_df['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Christmas : December 25 2017
    hist_df['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)

    # New Year's Day : January 01 2018
    hist_df['New_Years_Day_2018'] = (pd.to_datetime('2018-01-01') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Good Friday: March 30 2018
    # hist_df['Good_Friday_2018'] = (pd.to_datetime('2018-03-30') - hist_df['purchase_date']).dt.days.apply(
    #     lambda x: x if x > 0 and x < 10 else 0)
    # Tiradentes: April 21 2018
    # hist_df['Tiradentes_Day_2018'] = (pd.to_datetime('2018-04-21') - hist_df['purchase_date']).dt.days.apply(
    #     lambda x: x if x > 0 and x < 10 else 0)
    # Mothers Day: May 13 2018
    hist_df['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)

    hist_df['month_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days) // 30
    hist_df['month_diff'] += hist_df['month_lag']

    # additional features
    hist_df['duration'] = hist_df['purchase_amount'] * hist_df['month_diff']
    hist_df['amount_month_ratio'] = hist_df['purchase_amount'] / hist_df['month_diff']

    # reduce memory usage
    hist_df = utils.reduce_mem_usage(hist_df)

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
    aggs['month_diff'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['authorized_flag'] = ['mean', 'sum', 'min', 'max', 'var']
    # aggs['weekend'] = ['mean']  # overwrite
    # aggs['weekday'] = ['mean']  # overwrite
    # aggs['day'] = ['nunique', 'mean', 'min']  # overwrite
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size', 'count']
    aggs['price'] = ['sum', 'mean', 'max', 'min', 'var']

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
        hist_df[col + '_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
        hist_df[col + '_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
        hist_df[col + '_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
        hist_df[col + '_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col + '_mean'] = ['mean']

    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['hist_' + c for c in hist_df.columns]

    hist_df['hist_purchase_date_diff'] = (hist_df['hist_purchase_date_max']-hist_df['hist_purchase_date_min']).dt.days
    hist_df['hist_purchase_date_average'] = hist_df['hist_purchase_date_diff']/hist_df['hist_card_id_size']
    hist_df['hist_purchase_date_uptonow'] = (datetime.datetime.today()-hist_df['hist_purchase_date_max']).dt.days
    hist_df['hist_purchase_date_uptomin'] = (datetime.datetime.today()-hist_df['hist_purchase_date_min']).dt.days

    # reduce memory usage
    hist_df = utils.reduce_mem_usage(hist_df)

    return hist_df

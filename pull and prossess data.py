# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:03:53 2022

@author: alexp
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from matplotlib import pyplot as plt
import time


def pull_data(year, start, end, min_visits=10):
    '''
    Imports the Alphamega database classes used for quering the database.
    Then creates the sql query using the start and end date.
    Finally it pulls the data from the database
    
    Note: This will only work if using a Alphamega VPN.

    Parameters
    ----------
    year : int
        Year you want the data from.
    start : str
        Start of date. Date Format: 'YYYY-MM-DD'
    end : str
        End of date. Date Format: 'YYYY-MM-DD'
    min_visits: int
        Min number of visits per item customer pair.
        
    Returns
    -------
    df : Pandas DataFrame
        Dataframe that has every customer cat pair within the date range greater than the min_visits

    '''

    # Adds path where alpha
    sys.path.insert(1, 'C:/Users/alexp/Desktop/Work/Other/Useful/alphamega_db_classes')
    import alphamega_db_classes as adb

    if year % 2 == 0:
        year1 = year
        year2 = year + 1
    else:
        year1 = year - 1
        year2 = year
    year_str = f'{year1}_{year2}'
    db = adb.insight_sql_querier("Ins_AMe")
    db.open_conn()
    query = (f"""
        SELECT
            Cus_CardNo, 
            Pos_TimeDate , 
            sum([Pos_Quantity]) as qty , 
            [ICa_3DCode] AS  cat
        FROM [Ins_AMe].[dbo].[Ins_PosTransactions{year_str}]
        LEFT JOIN [Ins_Item] ON [Ins_Item].Itm_Code = [Ins_PosTransactions{year_str}].Itm_Code
            WHERE Pos_TimeDate >= '{start}' and  Pos_TimeDate <= '{end}' AND
            concat(ICa_3DCode, Cus_CardNo) IN ( 
                SELECT concat(ICa_3DCode, Cus_CardNo)
                FROM [Ins_AMe].[dbo].[Ins_PosTransactions{year_str}]
			 	LEFT JOIN [Ins_Item] ON [Ins_Item].Itm_Code = [Ins_PosTransactions{year_str}].Itm_Code
                WHERE Pos_TimeDate >= '{start}' and  Pos_TimeDate <= '{end}'
                GROUP BY ICa_3DCode, Cus_CardNo
                HAVING Cus_CardNo <> '' AND count(distinct Pos_TimeDate) >= {min_visits} 
                )
        GROUP BY ICa_3DCode, Cus_CardNo, Pos_TimeDate
        HAVING Cus_CardNo <> '' and  ICa_3DCode<> '969'             
             
        """)

    db.select_db_data(query)
    db.close_conn()
    df = db.get_data().copy()
    return df


def filter_cat_by_amount_cust(df, min_cust=10):
    '''
    Filters out categories where the number of customers is less than
    the min_cust.

    Parameters
    ----------
    df : Pandas Dataframe
        Data frame with cat and customers columns.
    min_cust : TYPE, optional
        The min number of customers who purchased each category. The default is 20.

    Returns
    -------
    df : Pandas Dataframe
        Same Dataframe as input but with only cats with enough customer purchases

    '''
    # groups the dataframe by cat and gets the unique count of other columns
    df_grouped = df.groupby(["Cus_CardNo", "cat"]).nunique()
    df_grouped = df_grouped.reset_index()
    # filters the group df by the min_cust and save the index of cat ids
    cats_to_keep = df_grouped[df_grouped.Pos_TimeDate > min_cust]
    # filter the main df using the cats_to_keep
    df = cats_to_keep[["Cus_CardNo", "cat"]].merge(df, how="left", left_on=["Cus_CardNo", "cat"],
                                                   right_on=["Cus_CardNo", "cat"])
    return df


def create_gap_df(df):
    '''
    Calculates the gap between each customer's purchase of a category and the previous purchase of that category.
    This is done by sorting the dataframe so every transaction of a given customer and category followed by the previous transaction.
    Then the dataframe is duplicated and the index is offset by -1. 
    The two dataframes are mergered on index, category and customer.
    This gives each row a column with data purchase and previous date purchased.
    Lastly these two columns are subtracted to get the number of days between purchases.
    
    Parameters
    ----------
    df : Pandas Dataframe
        Transactionsal data for eavery=

    Returns
    -------
    new_df : TYPE
        DESCRIPTION.

    '''
    # Sort values to have all transactions of the same customer and category are together and sorted in decending date order.
    df = df.sort_values(["Cus_CardNo", "cat", "Pos_TimeDate"], ascending=[True, True, False])
    # reset index 
    df = df.reset_index(drop=True)
    # duplicated dataframe
    df_copy = df.copy()
    # subtract 1 from duplicated dataframe
    df_copy.index = df_copy.index - 1
    df_copy.rename(columns={"Pos_TimeDate": "prev_date", "qty": "prev_qty"}, inplace=True)

    # make index a column in both dfs so we can use it to merge
    df_copy["index_copy"] = df_copy.index
    df["index_copy"] = df.index

    # merging both df and its copy on cat cust and index to make new_df. 
    # new_df has two date colomns now and because index is shifted and df is sorted, 
    # date from old df will have be current date and date from copy will have previous date.
    new_df = df.merge(df_copy, how="inner", right_on=("cat", "Cus_CardNo", "index_copy"),
                      left_on=("cat", "Cus_CardNo", "index_copy"))

    # Subtrace current with previous date to get gap
    new_df["gap"] = new_df.Pos_TimeDate - new_df.prev_date
    # format to show gap in days.
    new_df.gap = new_df.gap.dt.days

    return new_df


def add_past_time_period(df, t, cols, value_list=True):
    """
    Adds a column to df. This column contains a list of factors from time period [current_date - t]
    This is done using the same method as add_gap: copy df, shift index by -t, add cols data from that
    Parameters
    ----------
    df : Pandas Dataframe
        Transactional data for every customer and category.
    t : int
        Number of days to look back.
    cols : list
        List of columns to where the data will be pulled from at time period [current_date - t]
    Returns
    -------
    df : pandas dataframe
        Dataframe with the past time period column added.
    """

    # adds copy of index that will be shifted by -t
    df["index_copy"] = df.index
    # copies df
    copy = df.copy()
    # shifts index by -t of copy.
    copy.index_copy = copy.index_copy - t
    columns = cols + ["cat", "Cus_CardNo", "index_copy"]
    copy = copy[columns]  # keep only target cols
    if value_list:
        copy[t] = list(copy[cols].values)
    else:
        for col in cols:
            copy[f"{col}_{t}"] = copy[col]
    copy = copy.drop(columns=cols)
    df = df.merge(copy, how='left', right_on=("cat", "Cus_CardNo", "index_copy"),
                  left_on=("cat", "Cus_CardNo", "index_copy"))
    return df


def preprocess_df(df, past_cols=["gap", "qty"], scale_cols=["gap", "qty"], scale="standard", scale_y=True,
                  value_list=True, scaler=None):
    """
    Function that contains all preprocessing steps for the dataframe.

    Steps:
    1. Keeps only the columns we want to use.
        - Cat and Cus_CardNo are used for splitting the data.
        - gap will be used as both the target and a feature in the sequence input.
    2. If qty is True, qty will be added to as a second feature in the sequence input.
    3. Next the input features will be scaled using either standard, minmax scaling or none.
    4. The target will be scaled using the chosen scaling method if scale_y is True.
    5. Adds a column for each time period containing a list of features from time period.
        - Time period range is [1 to using_past]
    6. Finally, the dataframe is shuffled.

    Parameters
    ----------
    df : Pandas Dataframe
        Transactional data for every customer and category.
    qty : bool
        If True, quantity will be added to sequence.
    scale : str
        If "standard", standard scaler will be used.
        If "minmax", minmax scaler will be used.
        If "None", no scaling will be done.
    scale_y : bool
        If True, y will be scaled. Using scaler chosen by scale parameter.

    Returns
    -------
    df : Pandas Dataframe
        Dataframe with preprocessed data.
    """
    # list of columns we want to keep
    cols = ["cat", "Cus_CardNo", "Pos_TimeDate"] + past_cols
    # keep only columns we want to use
    df = df[cols]
    # creates separate column for target we want to predict (next gap).
    df["y"] = df.gap
    df.dropna(inplace=True)
    # scale input features
    if not scaler:
        if scale == "MinMax":
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df[scale_cols])
        elif scale == "standard":
            scaler = preprocessing.StandardScaler()
            scaler.fit(df[scale_cols])
    # Scale input features if scale is not None
    if scale != "None":
        df.loc[:, scale_cols] = scaler.transform(df.loc[:, scale_cols])
    # Scale target if scale_y is True
    if scale_y:
        df["y"] = df.gap
    df.dropna(inplace=True)
    # Loop through each time period and create a column for each time period containing a list of features from period.
    for t in reversed(range(1, using_past + 1)):
        df = add_past_time_period(df, t, past_cols, value_list=value_list)
    df.dropna(inplace=True)
    # group by customer and category and find max date for each customer and category
    grouped = df.groupby(["cat", "Cus_CardNo"])["Pos_TimeDate"].max()
    # reset index so we can merge with df
    grouped = grouped.reset_index()
    # merge with df, so we can keep only customer item pairs that have a max date.
    df = df.merge(grouped, how='inner', on=["cat", "Cus_CardNo", "Pos_TimeDate"])
    # shuffle dataframe
    df = df.sample(frac=1)
    return df


def x_y_split(df):
    df = df.dropna()
    x = df.drop(["cat", "Cus_CardNo", "gap", "index_copy", "y", "Pos_TimeDate","month_sine", "month_cos", "weekday_sine", "weekday_cos"], axis=1)

    if "qty" in x.columns:
        x = x.drop(["qty"], axis=1)

    y = df["y"].to_numpy()

    return x, y


def x_to_array(x):
    x = np.array(x.to_numpy().tolist()).astype('float32')
    return x


def df_date_to_month_weekday(df):
    month = df.Pos_TimeDate.dt.month
    weekday = df.Pos_TimeDate.dt.weekday
    df["month_sine"] = np.sin(month / 12 * 2 * np.pi)
    df["weekday_sine"] = np.sin(weekday / 7 * 2 * np.pi)
    df["month_cos"] = np.cos(month / 12 * 2 * np.pi)
    df["weekday_cos"] = np.cos(weekday / 7 * 2 * np.pi)
    return df

def scale_train_test(train,test, cols=[0,1], scaler=preprocessing.StandardScaler):
    scaler = scaler()
    for col in cols:
        scaler.fit(train[:, :, col])
        train[:, :, col] = scaler.transform(train[:, :, col])
        test[:, :, col] = scaler.transform(test[:, :, col])
    return train, test

# fixed paramaters
predict_next = 1
using_past = 10
epochs = 100
batch_size = 64
########
year = 2018
month1 = "01"
month2 = "03"

print(f"Pulling {month1} to {month2} of {year} data")
df = pull_data(year=year, start=f'{year}-{month1}-01', end=f'{year+1}-{month2}-30')
# removing blank category
df = df[df.cat != '$  ']
print("adding gap")
df = create_gap_df(df)
df = df_date_to_month_weekday(df)
df = preprocess_df(df,
                   scale="None",
                   scale_y=False,
                   value_list=True,
                   past_cols=["gap", "qty", "month_sine", "weekday_sine", "month_cos", "weekday_cos"])
x, y = x_y_split(df)
x = x_to_array(x)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

train_x, test_x = scale_train_test(train_x, test_x)

print("saving x and y")
np.save("data/X_train.npy", train_x)
np.save("data/X_test.npy", test_x)
np.save("data/y_train.npy", train_y)
np.save("data/y_test.npy", test_y)

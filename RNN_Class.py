# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:03:53 2022

@author: alex
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn import preprocessing
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional, RepeatVector, \
    TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt

print(device_lib.list_local_devices())


def pull_data(year, start, end, min_visits=10, group="cat"):
    """
    Imports the Alphamega database classes used for querying the database.
    Then creates the sql query using the start and end date.
    Finally, it pulls the data from the database

    Note: This will only work if using an Alphamega VPN.

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
    group: str
        Group column. Groups the data by the group column.
    Returns
    -------
    df : Pandas DataFrame
        Dataframe that has every customer cat pair within the date range greater than the min_visits


    """

    # Adds path where alpha
    sys.path.insert(1, 'C:/Users/alexp/Desktop/Work/Other/Useful/alphamega_db_classes')
    import alphamega_db_classes as adb

    # Transactional database splits data into two year tables.
    # If the year is even, year is the first year in the table name. Otherwise, year is the second year
    # This checks if the year is even or odd and sets the table name accordingly.
    if year % 2 == 0:
        year1 = year
        year2 = year + 1
    else:
        year1 = year - 1
        year2 = year
    year_str = f'{year1}_{year2}'
    # Creates the database object
    db = adb.insight_sql_querier("Ins_AMe")
    # Opens connection to the database
    db.open_conn()
    # Creates the sql query
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
                HAVING Cus_CardNo <> '' AND count(distinct Pos_TimeDate) >= 10 
                )
        GROUP BY ICa_3DCode, Cus_CardNo, Pos_TimeDate
        HAVING Cus_CardNo <> '' and  ICa_3DCode<> '969'             

        """)
    # Executes the query
    db.select_db_data(query)
    # Closes the connection to the database
    db.close_conn()
    # Saves the data to a dataframe
    df = db.get_data().copy()
    return df


def filter_cat_by_amount_cust(df, min_cust=10):
    """
    Filters out customers category pairs that have less than the min

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

    """
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
    """
    Calculates the gap between each customer's purchase of a category and the previous purchase of that category.
    This is done by sorting the dataframe so every transaction of a given customer and category followed by the
    previous transactions. Then the dataframe is duplicated and the index is offset by -1. The two dataframes are
    merged on index, category and customer. This gives each row a column with data purchase and previous date
    purchased. Lastly these two columns are subtracted to get the number of days between purchases.

    Parameters
    ----------
    df : Pandas Dataframe
        Transactional data for every customer and category.

    Returns
    -------
    new_df : Pandas Dataframe
        Dataframe with the gap between each customer's purchase of a category and the previous purchase of that category

    """
    # Sort values to have all transactions of the same customer and category are together
    # and sorted in descending date order.
    df = df.sort_values(["Cus_CardNo", "cat", "Pos_TimeDate"], ascending=[True, True, False])
    # reset index
    df = df.reset_index(drop=True)
    # duplicated dataframe
    df_copy = df.copy()
    # subtract 1 from duplicated dataframe
    df_copy.index = df_copy.index - 1
    # make index a column in both dfs, so we can use it to merge
    df_copy["index_copy"] = df_copy.index
    df["index_copy"] = df.index

    # merging both df and its copy on cat cust and index to make new_df.
    # new_df has two date columns now and because index is shifted and df is sorted,
    # date from old df will have been current date and date from copy will have previous date.
    new_df = df.merge(df_copy, how="inner", right_on=("cat", "Cus_CardNo", "index_copy"),
                      left_on=("cat", "Cus_CardNo", "index_copy"))

    # Subtract current with previous date to get gap
    new_df["gap"] = new_df.Pos_TimeDate_x - new_df.Pos_TimeDate_y
    # format to show gap in days.
    new_df.gap = new_df.gap.dt.days

    return new_df


def add_past_time_period(df, t, cols):
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
    copy[t] = list(copy[cols].values)
    copy = copy.drop(columns=cols)
    df = df.merge(copy, how='left', right_on=("cat", "Cus_CardNo", "index_copy"),
                  left_on=("cat", "Cus_CardNo", "index_copy"))
    return df


def preprocess_df(df, qty=False, scale="standard", scale_y=True):
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
    cols = ["cat", "Cus_CardNo", "gap"]
    # columns to add to sequence
    past_cols = ["gap"]
    # if qty is True, add qty to sequence and to columns to keep
    if qty:
        cols = cols + ["qty_x"]
        past_cols = past_cols + ["qty_x"]
    # keep only columns we want to use
    df = df[cols]
    # creates separate column for target we want to predict (next gap).
    df["y"] = df.gap
    df.dropna(inplace=True)
    # scale input features
    if scale == "MinMax":
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    elif scale == "standard":
        scaler = preprocessing.StandardScaler()
    # Scale input features if scale is not None
    if scale != "None":
        df.loc[:, past_cols] = scaler.fit_transform(df.loc[:, past_cols])
    # Scale target if scale_y is True
    if scale_y:
        df["y"] = df.gap
    df.dropna(inplace=True)
    # Loop through each time period and create a column for each time period containing a list of features from period.
    for t in reversed(range(1, using_past + 1)):
        df = add_past_time_period(df, t, past_cols)
    df.dropna(inplace=True)
    # shuffle dataframe
    df = df.sample(frac=1)
    return df


def RNN(layers, n_steps, n_features, dropout=0.2, lr=0.0003, l2=0.01):
    """
    Creates a recurrent neural network with specified layers, dropout, learning rate, and l2 regularization.

    Parameters
    ----------
    layers : list
        List of integers representing the number of nodes in each layer.
    n_steps : int
        Number of time steps in the input data.
    n_features : int
        Number of features in the input data.
    dropout : float
        Dropout rate.
    lr : float
        Learning rate.
    l2 : float
        L2 regularization.

    Returns
    -------
    model : Keras model
        Keras model.
    """
    # Creates a sequential model
    model = Sequential()
    # for
    for i in range(len(layers)):
        nodes = layers[i]
        if i + 1 == len(layers):
            if l2:
                model.add(Bidirectional(LSTM(nodes,
                                             activation='tanh',
                                             input_shape=(n_steps, n_features),
                                             kernel_regularizer=regularizers.l2(l2),
                                             bias_regularizer=regularizers.l2(l2)
                                             )
                                        )
                          )
            else:
                model.add(Bidirectional(LSTM(nodes,
                                             activation='tanh',
                                             input_shape=(n_steps, n_features),
                                             )
                                        )
                          )
        else:
            if l2:
                model.add(
                    Bidirectional(LSTM(nodes,
                                       activation='tanh',
                                       input_shape=(n_steps, n_features),
                                       return_sequences=True,
                                       kernel_regularizer=regularizers.l2(l2),
                                       bias_regularizer=regularizers.l2(l2)
                                       )
                                  )
                )
            else:
                model.add(
                    Bidirectional(LSTM(nodes,
                                       activation='tanh',
                                       input_shape=(n_steps, n_features),
                                       return_sequences=True,
                                       )
                                  )
                )
        if dropout:
            model.add(Dropout(dropout))

    model.add(Dense(6, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def plot_loss_graph(history):
    """
    Plot the loss graph.

    Parameters
    ----------
    history : keras.callbacks.History
        The history object returned by the model.fit() method.

    Returns
    -------
    None
    """
    axes = plt.axes()
    axes.plot(history.history['loss'])
    axes.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def gap_to_label(gap, group_list=[3,7,14,30,60]):
    prev_gap = 0
    array_list = []
    for i in range(len(group_list)):
        bool_array = (gap <= group_list[i]) * (gap > prev_gap)
        bool_array = np.array(bool_array, dtype=int)
        array_list.append(bool_array)
        prev_gap = group_list[i]
        if i == len(group_list) - 1:
            bool_array = gap > group_list[i]
            bool_array = np.array(bool_array, dtype=int)
            array_list.append(bool_array)
    array_list = np.array(array_list)
    return array_list.T


# fixed parameters
using_past = 10
epochs = 1000
batch_size = 256
filter_outlier = False
load_from_db = False
filter_top_n_cats = False
load_preprocessed_data = False
########a

train_x_array = np.load("data/X_train.npy")
train_y = np.load("data/y_train.npy")
train_y = gap_to_label(train_y)
train_limit = 7000000

# train_x_array = train_x_array[:train_limit]
# train_y = train_y[:train_limit]

test_x_array = np.load("data/X_test.npy")
test_y = np.load("data/y_test.npy")
test_limit = 20000
test_x_array = test_x_array[:test_limit]
test_y = test_y[:test_limit]
test_y = gap_to_label(test_y)

model = RNN([50,50], using_past, 2, dropout=0.5, lr=0.0003, l2=0.001)
earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=0.00001)

history = model.fit(train_x_array
                    , train_y
                    , epochs=epochs
                    , batch_size=batch_size
                    , verbose=1
                    , validation_data=(test_x_array, test_y)
                    , callbacks=[reduce_lr]
                    )

y_pred = model.predict(train_x_array)
y_pred_test = model.predict(test_x_array)



plot_loss_graph(history)

# plt.scatter(train_y, y_pred)
# plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:44:49 2022

@author: alexp
"""

import sys
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import r2_score
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/alexp/Desktop/Work/Other/Useful/alphamega_db_classes')

import alphamega_db_classes as adb

def pull_data():
    db = adb.insight_sql_querier("Ins_AMe")
    db.open_conn()
    query = (
        'SELECT  \n'
            'Itm_Code, Cus_CardNo, Pos_TimeDate , sum([Pos_Quantity]) as qty \n'
        'FROM [Ins_AMe].[dbo].[Ins_PosTransactions2020_2021] \n'
            'WHERE  MONTH(Pos_TimeDate) < 6 and YEAR(Pos_TimeDate) = 2021 and '
            'concat(Itm_Code, Cus_CardNo) IN ( \n'
                'SELECT concat(Itm_Code, Cus_CardNo) \n'
                'FROM [Ins_AMe].[dbo].[Ins_PosTransactions2020_2021] \n'
                'WHERE MONTH(Pos_TimeDate) < 6 and YEAR(Pos_TimeDate) = 2021 \n'
                'GROUP BY Itm_Code, Cus_CardNo \n'
                "HAVING Cus_CardNo <> '' AND count( distinct Pos_TimeDate) > 20 "
                ") \n"
        'GROUP BY Itm_Code, Cus_CardNo, Pos_TimeDate \n'
        "HAVING Cus_CardNo <> '' "
        )
    db.select_db_data(query)
    db.close_conn()
    df = db.get_data().copy()    
    return df

def create_gap_df(df):
    df = df.sort_values(["Cus_CardNo","Itm_Code","Pos_TimeDate"],ascending=[True,True,False])
    df = df.reset_index(drop=True)
    df_copy = df.copy()
    df_copy.index = df_copy.index-1
    df_copy["index_copy"] = df_copy.index
    df["index_copy"] = df.index
    new_df = df.merge(df_copy,how="inner",right_on=("Itm_Code", "Cus_CardNo","index_copy"),left_on=("Itm_Code", "Cus_CardNo","index_copy"))
    new_df["gap"] = new_df.Pos_TimeDate_x - new_df.Pos_TimeDate_y
    new_df.gap = new_df.gap.dt.days
    gap_df = new_df.groupby(["Itm_Code", "Cus_CardNo"]).agg({"gap": [np.mean, np.median,'count'],"Pos_TimeDate_x":["max"]})
    return gap_df, new_df

def add_past_time_period(df,time,col):
    df["index_copy"] = df.index
    copy = df.copy()
    copy.index_copy = copy.index_copy - time
    copy = copy[[col,"Itm_Code", "Cus_CardNo","index_copy"]]
    new_name = col+f'-{time}'
    copy = copy.rename({col:new_name},axis=1)
    df = df.merge(copy,how='left',right_on=("Itm_Code", "Cus_CardNo","index_copy"),left_on=("Itm_Code", "Cus_CardNo","index_copy"))
    return df 

df = pull_data()

gap_df, df = create_gap_df(df)
df_merge = df.merge(gap_df, how="left",left_on=["Itm_Code","Cus_CardNo"],right_on=["Itm_Code","Cus_CardNo"])

itm, cust = df_merge.sample(1)[["Itm_Code","Cus_CardNo"]].values[0]
test = df_merge[(df_merge.Itm_Code==itm) &( df_merge.Cus_CardNo == cust) ]
reg  = AutoReg(test.gap,9).fit()
reg.plot_predict()
test.gap.plot()

r2_score(test.gap[9:],reg.predict(start=9,end=len(test.gap)-1))

itm, cust = gap_df.index[10]
test = df[(df.Itm_Code == itm) & (df.Cus_CardNo == cust) ]
test3 = add_past_time_period(test,1,"gap")
test3 = add_past_time_period(test3,2,"gap")
test3 = add_past_time_period(test3,3,"gap")
test3 = add_past_time_period(test3,4,"gap")

test3 = test3[['gap',"gap-1",'gap-2','gap-3','gap-4']]
test3 = test3.dropna()
reg = LinearRegression()
reg.fit(test3[["gap-1",'gap-2','gap-3','gap-4']],test3.gap)
print(reg.score(test3[["gap-1",'gap-2','gap-3','gap-4']],test3.gap))

plt.plot(reg.predict(test3[["gap-1",'gap-2','gap-3','gap-4']]))
plt.plot(test3.gap.values)
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 14:01:10 2021

@author: alexp
"""
import sys
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
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
            'WHERE  MONTH(Pos_TimeDate) < 4 and YEAR(Pos_TimeDate) = 2021 and '
            'concat(Itm_Code, Cus_CardNo) IN ( \n'
                'SELECT concat(Itm_Code, Cus_CardNo) \n'
                'FROM [Ins_AMe].[dbo].[Ins_PosTransactions2020_2021] \n'
                'WHERE MONTH(Pos_TimeDate) < 4 and YEAR(Pos_TimeDate) = 2021 \n'
                'GROUP BY Itm_Code, Cus_CardNo \n'
                "HAVING Cus_CardNo <> '' AND count( distinct Pos_TimeDate) > 10 "
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
    df_copy.index = df_copy.index+1
    df_copy["index_copy"] = df_copy.index
    df["index_copy"] = df.index
    new_df = df.merge(df_copy,how="inner",right_on=("Itm_Code", "Cus_CardNo","index_copy"),left_on=("Itm_Code", "Cus_CardNo","index_copy"))
    new_df["gap"] = new_df.Pos_TimeDate_y - new_df.Pos_TimeDate_x
    new_df.gap = new_df.gap.dt.days
    gap_df = new_df.groupby(["Itm_Code", "Cus_CardNo"]).agg({"gap": [np.mean, np.median,'count'],"Pos_TimeDate_x":"max"})
    return gap_df, new_df


def poisson_prob(df):
    k = df["gap"]
    mu = df[("gap", "mean")]
    prob = stats.distributions.poisson.pmf(k,mu)
    return prob

def com_poisson(l,v,x):
    def Z(l,v):
        z=0
        for j in range(30):
            z+=l**j/(np.math.factorial(j)**v)
        return z 
    if type(x) == int or type(x) == float:
        return (l**x/(np.math.factorial(x))**v)*(1/Z(l,v))
    else:
        x = pd.Series(x)
        return (l**x/(x.apply(lambda y: np.math.factorial(y))**v))*(1/Z(l,v))
    
df = pull_data()

gap_df, df = create_gap_df(df)
df_merge = df.merge(gap_df, how="left",left_on=["Itm_Code","Cus_CardNo"],right_on=["Itm_Code","Cus_CardNo"])

probs_mean = stats.distributions.poisson.pmf(df_merge["gap"],df_merge[("gap","mean")])
probs_median = stats.distributions.poisson.pmf(df_merge["gap"],df_merge[("gap","median")])
comp05 =  com_poisson(df_merge[("gap","mean")],.5,df_merge["gap"])
comp07 =  com_poisson(df_merge[("gap","mean")],.7,df_merge["gap"])
comp1 =  com_poisson(df_merge[("gap","mean")],1,df_merge["gap"])

df_merge["probs_mean"] = probs_mean

df_merge["probs_median"] = probs_median

n=0
for i, c in gap_df.index:
    df_merge.loc[(df_merge.Itm_Code==i) & (df_merge.Cus_CardNo== c),"gap"].hist(density=True)
    mu =  gap_df.loc[(i, c),("gap","mean")]
    plt.plot(range(1,20),  stats.distributions.poisson.pmf(k=range(1,20),mu=mu))
    mu =  gap_df.loc[(i, c),("gap","median")]
    plt.plot(range(1,20),  stats.distributions.poisson.pmf(k=range(1,20),mu=mu))
    plt.plot(range(1,20),  com_poisson(mu,1,range(1,20)))

    
    n+=1
    plt.show()
    if n>10:
        break

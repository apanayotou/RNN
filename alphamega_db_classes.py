# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 08:44:02 2021

@author: alexp
"""
import pyodbc 
import pandas as pd
import adodbapi

    

class database_querier:
    def __init__(self):
        self.data = None
        self.conn = None
    
    def open_conn(self):
        conn = pyodbc.connect(self.conn_str, autocommit=True, readonly=True)          
        self.conn = conn
        print("Connected to database")
    
    def select_db_data(self, query):
        if self.conn == None:
            print("Not connected to DB. Use self.open_conn() first.")
        else:
            data = pd.io.sql.read_sql(query, self.conn)
            self.data = data
            print("Data retrevied. Use self.get_data")
    def alter_db_data(self, query):
        if self.conn == None:
            print("Not connected to DB. Use self.open_conn first.")
        else:
            self.conn.execute(query)
        
    def close_conn(self):
        self.conn.close()
        self.conn = None
        print("Connection closed")
    
    def get_data(self):
        return self.data
    
    def get_status(self):
        if self.conn==None:
            print("No connection")
        else:
            print("Connected")
            
    def insert_data(self, data_list, table, condition=None):
        data_list: data_value_list
        table: str
        conditon: str
        
        data, cols = data_list.make_sql_insert_str()
        
        if condition == None:
            where = ''
            condition = ''
        else:
            where = 'WHERE'
        
        if type(data) != str:
            data = str(data)
            data = data[1:][:-1]
        q = f"INSERT INTO {table} {cols} VALUES {data} {where} {condition}"
        print (q)
        self.alter_db_data(q)
        
    
    def update_data(self, data_value_list, table, condition):

        data = data_value_list.make_sql_update_str()
        condition = condition.make_sql_update_str()
        if condition == None:
            where = ''
            condition = ''
        else:
            where = 'WHERE'
            
        q = f"UPDATE {table} SET {data} {where} {condition}"
        print(q)
        self.alter_db_data(q)          
            
class sql_querier(database_querier):
    def __init__(self, driver, server, database):
        super().__init__()
        self.driver = driver
        self.server = server
        self.database = database  
        self.conn_str = f'Driver={driver}; Server={server};Database={database};Trusted_Connection=yes;'
                             
class aurora_querier(database_querier):
    def __init__(self, driver, system, library, uid, pwd):
        super().__init__()
        self.driver = driver
        self.system = system
        self.library = library
        self.uid = uid
        self.pwd = pwd
        self.conn_str = f"DRIVER={driver};SYSTEM={system};LIBRARY={library};UID={uid};PWD={pwd}"

class cube_querier(database_querier):
    def __init__(self, provider, security, security_info, catalog, data_source, mdx_comp, safety_options, mdx_missing ):
        super().__init__()
        self.provider = provider
        self.security = security
        self.security_info = security_info
        self.catalog = catalog
        self.data_source = data_source
        self.mdx_comp = mdx_comp
        self.safety_options = safety_options
        self.mdx_missing = mdx_missing
        self.conn_str= f'''Provider={self.provider};
                                Integrated Security={self.security};
                                Persist Security Info={self.security_info};
                                Initial Catalog={self.catalog};
                                Data Source={data_source};
                                MDX Compatibility={self.mdx_comp};
                                Safety Options={self.safety_options};
                                MDX Missing Member Mode={self.mdx_missing}'''
    def open_conn(self):
        conn = adodbapi.connect(self.conn_str)
        self.conn = conn
        

    def select_db_data(self,query):
        crsr = self.conn.cursor()
        crsr.execute(query)
        data = crsr.fetchall()   
        data = data.ado_results
        return data
    
    
    def alter_db_data(self, query):
        print("Cant alter data in cube.")
    
class mfile_querier(sql_querier):
    def __init__(self, database):
        super().__init__("SQL Server", "ACRMFILESDBSRV", database)  
        
class test_serv_querier(sql_querier):
    def __init__(self, database):
        super().__init__("SQL Server", "ACRANALYTESTSRV", database)   

class insight_sql_querier(sql_querier):
    def __init__(self, database):
        super().__init__("SQL Server", "insight", database)   

class aurora_querier_main(aurora_querier):
    def __init__(self):
        super().__init__("iSeries Access ODBC Driver", "10.20.0.1", "AULCAPF3","mfiles","mfiles")   

class insight_cube_querier(cube_querier):
    def __init__(self):
        super().__init__("MSOLAP", "SSPI", "True","InsAMe","Insight","1","2","Error")   
    
    def get_data_simple(measure,item,promo,start_date, end_date):
        pass
        
class data_value:
    def __init__(self,column,value,dtype=None,logic="=" ):
        self.column = column
        if dtype == None:
            dtype = type(value)
        else:
            value = dtype(value)
        self.dtype = dtype        
        self.value = value
        self.logic = logic
    def get_column(self):
        return self.column
    def get_value(self):
        return self.value
    def get_logic(self):
        return self.logic
    def get_dtype(self):
        return self.dtype
    
    def get_sql_update_str(self):
        if self.dtype == str:
            string = f"{self.get_column()} {self.get_logic()} '{self.get_value()}'"
        else:
            string = f"{self.get_column()} {self.get_logic()} {self.get_value()}"
        return string
    
    def __repr__(self):
        return f"{self.get_column()} : {self.get_value()}"

class data_value_list:
    def __init__(self):
        self.value_list = []
        
    def add_value(self,value):
        self.value_list.append(value)
        
    def get_value_list(self):
        return self.value_list
    
    def make_sql_update_str(self):
        sql_str = ''
        for value_obj in self.get_value_list():
            sql_str += f"{value_obj.get_sql_update_str()},"
        sql_str = sql_str[:-1]
        return f"{sql_str}"
    def make_sql_insert_str(self):
        val_str = '' 
        col_str = '' 
        for value_obj in self.get_value_list():
            if value_obj.get_dtype() == str:
                val_str += f"'{value_obj.get_value()}',"
            else:
                val_str += f"{value_obj.get_value()},"    
            col_str += f"{value_obj.get_column()},"
        val_str, col_str = val_str[:-1], col_str[:-1]
        return f"({val_str})", f"({col_str})"
    
    def multi_add_value(self, my_list):
        for item in my_list:
            val = data_value(item[0], item[1])
            self.add_value(val)

class data_row(pd.Series):
    def set_update_key(self,key):
        self.key = key
    def make_sql_insert_str(self):
        vals = self.map(lambda x: f"'{str(x)}'")
        col_str = ", ".join(self.index)
        val_str = ", ".join(vals)
        return(col_str, val_str)        
    
    def make_sql_update_str(self):
        sql_list = []
        for i in self.index:
            if type(self.loc[i]) == str:
                sql_list.append(f"{i} = '{self.loc[i]}'")
            else:
                sql_list.append(f"{i} = {self.loc[i]}")
        sql_str = ", ".join(sql_list)
        return sql_str


if __name__ == "__main__":
    test = test_serv_querier("Test")
    test.open_conn()
    
    update_list = data_value_list()
    update_list.multi_add_value([("Address","other"),("City","Larnica")])
    conditon_list = data_value_list()
    conditon_list.multi_add_value([("FirstName","Donna")])
    
    test.update_data(update_list, "Persons", conditon_list)
    
    



    
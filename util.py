import datetime
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 


def pre_processing(df):
    '''
	Definition:
		This function pre processing the data
	args:
		data to be pre processed
	returns:
		data pre processed
		
    '''	
	
    
    if("Unnamed: 0" in df.columns):
        df.drop(columns=["Unnamed: 0"],inplace=True)
        
    if("LNR" in df.columns):
        df.set_index("LNR",inplace=True)
        
    #TOO MANY VALUES
    if("D19_LETZTER_KAUF_BRANCHE" in df.columns):
        df.drop(columns = ["D19_LETZTER_KAUF_BRANCHE"],inplace=True)
    
    def map_year(x):
        x = str(x)
        year = x.split("-")[0]
        return year
    
    
        
    def map_cameo_deu(x):
        letter = x[1]
        letter_dict = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
        return letter_dict[letter]
        
    df["CAMEO_DEU_2015_LETTER"] = df["CAMEO_DEU_2015"].apply(lambda x: np.nan if (x=="XX" or x=="X" or x=="" or x==" " or str(x)=="nan") else map_cameo_deu(x)).astype(float)
    df["CAMEO_DEUG_2015"] = df["CAMEO_DEUG_2015"].apply(lambda x: np.nan if (x=="XX" or x=="X" or x=="" or x==" " or str(x)=="nan") else x).astype(float)
    df["CAMEO_INTL_2015"] = df["CAMEO_INTL_2015"].apply(lambda x: np.nan if (x=="XX" or x=="X" or x=="" or x==" " or str(x)=="nan") else x).astype(float)
    df["OST_WEST_KZ"] = df["OST_WEST_KZ"].map({"W":0,"O":1,np.nan:np.nan}).astype(float)
    df["EINGEFUEGT_AM"] = df["EINGEFUEGT_AM"].apply(lambda x: map_year(x) if str(x)!="nan" else map_year(x)).astype(float)
    
    
    return df


def feature_eng(df):
    '''
	Definition:
		This function creates and transforms features
	args:
		dataframe to be processed
	returns:
		dataframe with new and transformed features 
		
    '''	
    #WOHNLAGE
    area_dict = {1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 7.0:1, 8.0:1}
    #WOHNLAGE
    quality_dict = {1.0:1, 2.0:1, 3.0:2, 4.0:3, 5.0:3,7:-1,8:-1}
     
    df["WOHNLAGE_URBAN_OR_RURAL"] = df["WOHNLAGE"].map(area_dict).astype(float)
    df["WOHNLAGE_QUALITY"] = df["WOHNLAGE"].map(quality_dict).astype(float)
       
    #LP_STATUS_GROB
    social_status_dict = {1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:3,9:3,10:4}
    df["LP_STATUS_GROB"] = df["LP_STATUS_GROB"].map(social_status_dict).astype(float)

    #LP_FAMILIE_GROB
    family_size = {1:0,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}
    df["LP_FAMILIE_GROB"] = df["LP_FAMILIE_GROB"].map(family_size).astype(float)
    
    

    transactions_mail_order_array = ["D19_VERSAND_ONLINE_QUOTE_12",
                                     "D19_BANKEN_ONLINE_QUOTE_12",
                                     "D19_GESAMT_ONLINE_QUOTE_12"]

    transactions_mail_order = {0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:2,8:2,9:2,10:3}
    
    for tmo in transactions_mail_order_array:
        df[tmo] = df[tmo].map(transactions_mail_order).astype(float)



    transactions_online_array = ["D19_BANKEN_DATUM",
    "D19_BANKEN_OFFLINE_DATUM",
    "D19_BANKEN_ONLINE_DATUM",
    "D19_GESAMT_DATUM",
    "D19_GESAMT_OFFLINE_DATUM",
    "D19_GESAMT_ONLINE_DATUM",
    "D19_TELKO_DATUM",
    "D19_TELKO_OFFLINE_DATUM",
    "D19_TELKO_ONLINE_DATUM",
    "D19_VERSAND_DATUM",
    "D19_VERSAND_OFFLINE_DATUM",
    "D19_VERSAND_ONLINE_DATUM"]

    transactions_online = {1:1,2:1,3:1,4:2,5:2,6:3,7:3,8:3,9:3,10:0} 
    for to in transactions_online_array:
        df[to] = df[to].map(transactions_online).astype(float)

    


    transactions_activity_array = ["D19_VERSI_ANZ_12",
    "D19_VERSI_ANZ_24",
    "D19_BANKEN_ANZ_12",
    "D19_BANKEN_ANZ_24",
    "D19_GESAMT_ANZ_12",
    "D19_GESAMT_ANZ_24",
    "D19_TELKO_ANZ_12",
    "D19_TELKO_ANZ_24",
    "D19_VERSAND_ANZ_12",
    "D19_VERSAND_ANZ_24"]


    transactions_activity = {0:0,1:1,2:1,3:2,4:2,5:3,6:3}
    
    
    for ta in transactions_activity_array:
        df[ta] = df[ta].map(transactions_activity).astype(float)

    def map_wealth(wealth):
        if(str(wealth)=="nan" or wealth == None):
            return np.nan
        if(wealth>=11 and wealth<=15):
            return 4
        if(wealth>=21 and wealth<=25):
            return 3
        if(wealth>=31 and wealth<=35):
            return 2
        if(wealth>=41 and wealth<=45):
            return 1
        if(wealth>=51 and wealth<=55):
            return 0

    def map_movement(x):
        if(x in [1,3,5,8,10,12,14]):
            return 1
        if(x in [2,4,6,7,9,11,13,15]):
            return 0
        else:
            return np.nan

    def map_generation():
        return {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:1,9:1,10:1,12:1,13:1,14:2,15:2}

    def map_life_stating(x):
        if(str(x)=="nan" or x==None):
            return np.nan

        return int(x)%10

    def map_status(x):
        if(str(x)=="nan" or x == None):
            return np.nan
        if(x>=1 and x<=2):
            return 4
        if(x>=3 and x<=5):
            return 3
        if(x>=6 and x<=7):
            return 2
        if(x>7):
            return 1
        
        
    df["CAMEO_INTL_2015_WEALTH"] = df["CAMEO_INTL_2015"].apply(lambda x: map_wealth(x)).astype(float)
    
    df["CAMEO_INTL_2015_LIFE_STATING"] = df["CAMEO_INTL_2015"].apply(lambda x: map_life_stating(x)).astype(float)
    
    df["CAMEO_DEUG_2015_WEALTH_STATUS"] = df["CAMEO_DEUG_2015"].apply(lambda x: map_status(x)).astype(float)

    
    df["PRAEGENDE_JUGENDJAHRE_GENERATION"] = df["PRAEGENDE_JUGENDJAHRE"].map(map_generation())
    
    df["PRAEGENDE_JUGENDJAHRE_MOVEMENT"] = df["PRAEGENDE_JUGENDJAHRE"].apply(lambda x: map_movement(x))

   
    
    
    life_age = {1: 1, 2: 2, 3: 1,
              4: 2, 5: 3, 6: 4,
              7: 3, 8: 4, 9: 2,
              10: 2, 11: 3, 12: 4,
              13: 3, 14: 1, 15: 3,
              16: 3, 17: 2, 18: 1,
              19: 3, 20: 3, 21: 2,
              22: 2, 23: 2, 24: 2,
              25: 2, 26: 2, 27: 2,
              28: 2, 29: 1, 30: 1,
              31: 3, 32: 3, 33: 1,
              34: 1, 35: 1, 36: 3,
              37: 3, 38: 4, 39: 2,
              40: 4}

    wealt_scale = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1,
              7: 2, 8: 2, 9: 2, 10: 3, 11: 2,
              12: 2, 13: 4, 14: 2, 15: 1, 16: 2,
              17: 2, 18: 3, 19: 3, 20: 4, 21: 1,
              22: 2, 23: 3, 24: 1, 25: 2, 26: 2,
              27: 2, 28: 4, 29: 1, 30: 2, 31: 1,
              32: 2, 33: 2, 34: 2, 35: 4, 36: 2,
              37: 2, 38: 2, 39: 4, 40: 4}
    
    df['LP_LEBENSPHASE_FEIN_AGE'] = df['LP_LEBENSPHASE_FEIN'].map(life_age)
    df['LP_LEBENSPHASE_FEIN_WEALTH'] = df['LP_LEBENSPHASE_FEIN'].map(wealt_scale)

      
    #FEATURES
    df.drop(columns=["WOHNLAGE","PRAEGENDE_JUGENDJAHRE","LP_LEBENSPHASE_FEIN","CAMEO_DEUG_2015","CAMEO_INTL_2015","CAMEO_DEU_2015"],inplace=True)
        
    return df

def map_to_unkown(df):
    '''
	Definition:
		This function maps categoricla features with representative NaN values, acordding to unkown_values dictionary
	args:
		dataframe to be processed
	returns:
		dataframe mapped 
		
    '''	
    df_unknow_values = pd.read_csv("unknow_values.csv",sep=";")
    mapping_unkown_values = {}
    for index,row in df_unknow_values.iterrows():
        values_splitted =  row["Value"].replace(" ","").split(",")
        values_splitted =list(map(int, values_splitted))
        mapping_unkown_values[row["Attribute"]] = values_splitted
    
    
    
    data = []
    count = 0
    for index, row in df.iterrows():
        if(count%1000==0):
            print(count," rows processed")
        count+=1
        new_row = []
        for column in row.keys():
            new_value = row[column]
            if column in mapping_unkown_values.keys():
                if row[column] in mapping_unkown_values[column]:
                    new_value = np.nan
                else:
                    new_value = row[column]
            new_row.append(new_value)
        data.append(new_row)
    return pd.DataFrame(data,columns=df.columns)   

def pre_processing_customers(customers):
    '''
	Definition:
		This function eliminates some features of custumers
	args:
		dataframe to be processed
	returns:
		dataframe without some features
		
    '''	
    return customers.drop(columns=['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP'],inplace=True)   


def transform_and_scale(df,imputer_strategy="most_frequent"):    
    '''
	Definition:
		This function imput null values and scale all falues
	args:
		dataframe to be processed
	returns:
		dataframe inputted and scaled
		
    '''	
    df = df.astype(float)
    imputer = SimpleImputer(missing_values=np.nan,strategy=imputer_strategy)
    df_imputed = imputer.fit_transform(df)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_imputed)
    return df_scaled    




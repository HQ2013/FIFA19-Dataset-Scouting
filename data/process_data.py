"""
Data Processing Script
For Udacity Data Scientist Nanodegree Program Project: Capstone Project-FIFA19 Scouting

Usage:
> python process_data.py data.csv
Arguments:
    1) Input  File: data.csv         - CSV file containing FIFA19 players' data
    2) Output File: cleaned_data.csv - processed data
"""

# import libraries
import sys
import time
import math
import calendar
import numpy  as np
import pandas as pd
from math    import sqrt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

#############################################################
def load_data(csv_filepath):
    """
    Load Data function:
    1. read input csv file containing FIFA19 players' data
    2. Assign this combined dataset to df
    
    Arguments:
    Input:
        data.csv -> CSV file containing FIFA19 players' data
    Output:
        df       -> Return data as Pandas DataFrame
    """
    df = pd.read_csv(csv_filepath)
    del df['Unnamed: 0']
    return df 

#############################################################
def clean_data(df):
    """Clean FIFA19 data for a visualizaiton dashboard
    Clean Data function
    1. Check & Impute Missing Values
    2. Convert string/date values into numbers
    3. Dealing with categorical features
    4. Normalization of feature "Height", "Weight"
    5. Combine Position Rating features with average value, this is for Radar Plotting

    Arguments:
        df - raw     data Pandas DataFrame
    Outputs:
        df - cleaned data Pandas DataFrame
    """
    #Keep only the rows with at least half non-NA values.
    df.dropna(thresh=14, inplace=True)
    
    #Drop Column "Loaned From"
    df.drop('Loaned From', axis=1,inplace=True)
    
    #For GK, fill missing value at other positions as "0+0"
    Position_List = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM',
                     'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
    df.loc[df.Position == 'GK', Position_List] = '0+0'
    
    #Fill Missing Value in "Release Clause" with "€0"
    df['Release Clause'].fillna(value='€0', inplace=True)
    
    #Drop Column "Joined"
    df.drop('Joined', axis=1,inplace=True)
    
    #Fill Missing Value in "Club" with "Free Contract"
    df['Club'].fillna(value='Free Contract', inplace=True)
    
    #Fill Missing Value in "Contract Valid Until" with today's date
    df['Contract Valid Until'].fillna(value=time.strftime("%d-%m-%Y", time.localtime()), inplace=True)
    
    #Drop the rows still with Missing Values(only 12 rows with lots of feature missing
    df.dropna(axis=0, inplace=True)
    
    #Create New Value_Number_K, Wage_Number_K, ReleaseClause_Number_K column to store numerical type Value info
    df['Value_Number_K'] = df['Value'].map(lambda x: str2number(x)/1000)
    df['Wage_Number_K']  = df['Wage'].map(lambda x: str2number(x)/1000)
    df['ReleaseClause_Number_K'] = df['Release Clause'].map(lambda x: str2number(x)/1000)

    #Create New Contract_Remaining_Month_Number column to store numerical type Remaining Contract info
    df['Contract_Remaining_Month_Number'] = df['Contract Valid Until'].map(lambda x: date2monthsnumber(x))

    #Calculate the final rating and the total increment
    Increment_List= []
    for pos in Position_List:
        new_column_name_Final     = pos+"_Final"
        new_column_name_Increment = pos+"_Increment"
        df[new_column_name_Final]     = df[pos].map(lambda x: rating2finalandincrement(str(x),"final"))
        df[new_column_name_Increment] = df[pos].map(lambda x: rating2finalandincrement(str(x),"increment"))
    Increment_List.append(new_column_name_Increment)
    
    df['Total_Increment'] = df[Increment_List].sum(axis=1)
    df.drop(Increment_List, axis=1,inplace=True)
    
    # One-hot encode the feature: "Nationality", "Club", "Work Rate", "Body Type", "Preferred Foot", and "Position"
    le = LabelEncoder()
    df['Nationality_onehot_encode']   = le.fit_transform(df['Nationality'])
    df['Club_onehot_encode']          = le.fit_transform(df['Club'])
    df['WorkRate_onehot_encode']      = le.fit_transform(df['Work Rate'])
    df['BodyType_onehot_encode']      = le.fit_transform(df['Body Type'])
    df['PreferredFoot_onehot_encode'] = le.fit_transform(df['Preferred Foot'])
    df['Position_onehot_encode']      = le.fit_transform(df['Position'])
    
    # Normalization of feature "Height", "Weight" using min max normalization
    df['Height_float'] = df['Height'].map(lambda x: convertHeightWeight2floatnumber(str(x),"Height"))
    df['Weight_float'] = df['Weight'].map(lambda x: convertHeightWeight2floatnumber(str(x),"Weight"))
    df['Height_Normalized'] = (df['Height_float'] - df['Height_float'].min())/(df['Height_float'].max() - df['Height_float'].min())
    df['Weight_Normalized'] = (df['Weight_float'] - df['Weight_float'].min())/(df['Weight_float'].max() - df['Weight_float'].min())
    
    #Combine Position Rating with average value
    PAC_List = ['Acceleration','SprintSpeed','Agility']
    SHO_List = ['ShotPower','LongShots','Finishing','FKAccuracy','HeadingAccuracy','Penalties','Curve']
    PAS_List = ['ShortPassing','LongPassing','Crossing','FKAccuracy','HeadingAccuracy','Curve','Vision']
    DRI_List = ['Dribbling','BallControl','SprintSpeed']
    DEF_List = ['Interceptions','StandingTackle','SlidingTackle','Positioning','Volleys','Marking','Vision']
    PHY_List = ['Stamina','Strength','Jumping','Balance','Aggression','Reactions']

    DIV_List = ['GKDiving']
    HAN_List = ['GKHandling']
    KIC_List = ['GKKicking']
    REF_List = ['GKReflexes']
    SPD_List = ['Agility','SprintSpeed','Acceleration']
    POS_List = ['GKPositioning']

    df['PAC'] = df[PAC_List].mean(axis=1)
    df['SHO'] = df[SHO_List].mean(axis=1)
    df['PAS'] = df[PAS_List].mean(axis=1)
    df['DRI'] = df[DRI_List].mean(axis=1)
    df['DEF'] = df[DEF_List].mean(axis=1)
    df['PHY'] = df[PHY_List].mean(axis=1)
    df['DIV'] = df[DIV_List].mean(axis=1)
    df['HAN'] = df[HAN_List].mean(axis=1)
    df['KIC'] = df[KIC_List].mean(axis=1)
    df['REF'] = df[REF_List].mean(axis=1)
    df['SPD'] = df[SPD_List].mean(axis=1)
    df['POS'] = df[POS_List].mean(axis=1)
    
    # output clean csv file
    return df

#############################################################
# Supporting function to convert string values into numbers
def str2number(amount):
    """
    This function perform convertion from amount values in string type to float type numbers
    
    Parameter:
    amount(str): Amount values in string type with M & K as Abbreviation for Million and Thousands
    
    Returns:
    float: A float number represents the numerical value of the input parameter amount(str)
    """
    if amount[-1] == 'M':
        return float(amount[1:-1])*1000000
    elif amount[-1] == 'K':
        return float(amount[1:-1])*1000
    else:
        return float(amount[1:])

#############################################################
# Supporting function to convert two types of date values into remaining months numbers from the current date
def date2monthsnumber(date):
    """
    This function perform convertion from two types of date values into remaining months numbers from the current date
    Type 1: Just year number YYYY.            For example, 2021
    Type 2: Complete date format DD-MMM-YYYY. For example, 28-Jul-2017
    
    Parameter:
    date(str): Two types of date representation as stated above
    
    Returns:
    Numerical: A number represents the remaining months numbers from the current date
    """
    
    current_month = int(time.strftime("%m", time.localtime()))
    current_year  = int(time.strftime("%Y", time.localtime()))
    
    if (len(str(date)) == 12) or (len(str(date)) == 11):
        month,day,year = date.split(" ")
        contract_end_month = dict((v,k) for k,v in enumerate(calendar.month_abbr))[month]
        contract_end_year  = int(year)
        contract_remain_month = int((contract_end_year-current_year)*12 + (contract_end_month-current_month))
        if contract_remain_month < 0:
            return 0
        else:
            return contract_remain_month
        
    elif len(str(date)) ==  10:
        day,month,year = date.split("-")
        contract_end_month = int(month)
        contract_end_year  = int(year)
        contract_remain_month = int((contract_end_year-current_year)*12 + (contract_end_month-current_month))
        if contract_remain_month < 0:
            return 0
        else:
            return contract_remain_month

    elif len(str(date)) ==  4:
        contract_end_month = 12
        contract_end_year  = int(date)
        contract_remain_month = int((contract_end_year-current_year)*12 + (contract_end_month-current_month))   
        if contract_remain_month < 0:
            return 0
        else:
            return contract_remain_month

#############################################################
# Supporting function to convert str type Position Rating into final rating & increment as well
def rating2finalandincrement(rating,return_type):
    """
    This function perform convertion from str type Position Rating into final rating & increment as well
    
    Parameter:
    rating(str):      Position Rating Expression
    return_type(str): Type of the return value
    
    Returns:
    Numerical: For return_type == "final",    returns a number represents the final     rating of this player at this position
               For return_type == "increment, returns a number represents the increment rating of this player at this position
    """
    final_rating,increment_rating = rating.split("+")
    
    if return_type == "final":
        return int(final_rating) + int(increment_rating)
    elif return_type == "increment":
        return int(increment_rating)        

#############################################################
# Supporting function to convert Height & Weight to float number
def convertHeightWeight2floatnumber(data,type):
    """
    This function perform convertion from str type Height & Weight into floating number type
    
    Parameter:
    data(str): Height/Weight
    type(str): Type of the data, Height or Weight
    
    Returns:
    Numerical: floating number of Height/Weight
    """
   
    if type == "Height":
        a,b=data.split("'")
        return float(a)+float(b)/10
    elif type == "Weight":
        c=data[0:-3]
        return float(c)        

#############################################################
def save_data(df, database_filename):
    """
    Save Data function
    1. Save the clean dataset into csv file
    
    Arguments:
        df                - Cleaned data Pandas DataFrame
        database_filename - csv file destination path
    """
    df.to_csv(database_filename)
    pass

#############################################################
def main():
    """
    Data Processing Main function
    
    This function implement the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and pre-processing
        3) Data loading to csv file
    """
    print(sys.argv)

    if len(sys.argv) == 3:

        csv_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    Players: {}\n'.format(csv_filepath))
        df = load_data(csv_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the FIFA19 csv dataset as the'\
              'first argument respectively, as well as the filepath of the'\
              'database to save the cleaned data to as the second argument.'\
              '\n\nExample: python process_data.py data.csv cleaned_data.csv')

#############################################################
if __name__ == '__main__':
    main()
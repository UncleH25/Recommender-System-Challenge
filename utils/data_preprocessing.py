#imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#Function to preprocess the kaggle data
def preprocess_kaggle_data(df):
    """
    Basic preprocessing for Kaggle dataset:
    - Encode CustomerID and StockCode
    - Return modified DataFrame and encoders
    """

    #Make a copy of the kaggle data
    df = df.copy()

    #Encode CustomerID and StockCode
    kaggle_user_encoder = LabelEncoder()
    kaggle_item_encoder = LabelEncoder()

    #Encode the CustomerID and StockCode as integers and store them in their new columns
    df['user_id'] = kaggle_user_encoder.fit_transform(df['CustomerID'])
    df['item_id'] = kaggle_item_encoder.fit_transform(df['StockCode'])

    return df, kaggle_user_encoder, kaggle_item_encoder

#Function to preprocess the fnb data
def preprocess_fnb_data(df):
    """
    Basic preprocessing for FNB dataset:
    - Encode idcol and item
    - Convert interaction to a numeric level (optional basic version)
    - Return modified DataFrame and encoders
    """

    #Make a copy of the fnb data
    df = df.copy()

    #Encode idcol (cutomer) and item
    fnb_user_encoder = LabelEncoder()
    fnb_item_encoder = LabelEncoder()

    #Encode the idcol and item as integers and store them in their new columns
    df['user_id'] = fnb_user_encoder.fit_transform(df['idcol'])
    df['item_id'] = fnb_item_encoder.fit_transform(df['item'])

    #Basic interaction score mapping
    interaction_map = {'DISPLAY': 1, 'CLICK': 2, 'CHECKOUT': 3}
    #If the interaction column is not in the dataframe, create it randomly
    if 'Interaction' not in df.columns:
        df['Interaction'] = np.random.choice(list(interaction_map.keys()), size=len(df))
    df['interaction_score'] = df['Interaction'].map(interaction_map)

    #Implicit score based on quantity
    df['interaction_score'] = df['Quantity']  

    return df, fnb_user_encoder, fnb_item_encoder


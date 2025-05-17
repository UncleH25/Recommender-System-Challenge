#imports
import pandas as pd

#Function to get the top N items for kaggle dataset
def get_top_kaggle_items(df, top_n=10):
    """
    Recommends top-N most popular products from Kaggle dataset
    based on total Quantity sold.
    """

    #Group item popularity by product code and sum the quantity sold and sort the values in descending order
    item_popularity = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
    #Get the top N items
    return item_popularity.head(top_n)

#Function to get the top N items for fnb dataset
def get_top_fnb_tems(df, top_n=10):
    """
    Recommends top-N most frequently CHECKOUTed items from FNB dataset.
    """

    #Get the top N items based on the CHECKOUT interaction
    df_checkout = df[df['interaction'] == 'CHECKOUT']
    #Count the number of CHECKOUT interactions for each item
    item_popularity = df_checkout['item'].value_counts().head(top_n)
    #Return the top N items
    return item_popularity
#imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Function for content-based filtering using TF-IDF
def build_content_matrix(df, content_col='item_descrip'):
    """
    Build TF-IDF matrix from item_descrip or item_type
    Returns:
    - tfidf_matrix: item × token matrix
    - item_indices: mapping of item_id to row index
    """

    #Initialize a TF-IDF vectorizer that removes English stop words
    tfidf = TfidfVectorizer(stop_words='english')
    #Fit the vectorizer on the content column (fill missing values with empty string) and transform to a TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(df[content_col].fillna(""))
    #Create a mapping from item_id to the row index in the TF-IDF matrix, ensuring no duplicates
    item_indices = pd.Series(df.index, index=df['item_id']).drop_duplicates()

    return tfidf_matrix, item_indices
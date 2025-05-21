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

#Function to get similar items based on cosine similarity
def get_similar_items(item_id, tfidf_matrix, item_indices, top_n=10):
    """
    Get top-N items most similar to the given item_id based on cosine similarity
    """

    #Get the row index of the given item_id in the TF-IDF matrix
    idx = item_indices[item_id]

    #Compute cosine similarity between the given item's vector and all item vectors
    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    #Pair each item index with its similarity score
    sim_scores = list(enumerate(cosine_sim))
    #Sort the items by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #Select the top_n most similar items, excluding the item itself (which is always first)
    top_items = sim_scores[1:top_n+1]

    return [(item_id, score) for item_id, score in top_items]
#imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Function for content-based filtering using TF-IDF
def build_content_matrix(df, content_col='item_descrip'):
    """
    Build a TF-IDF matrix from item_descrip or item_type
    Returns:
    - tfidf_matrix: item * token matrix
    - item_indices: mapping of item_id to row index
    """

    #Initialize a TF-IDF vectorizer that removes English stop words
    tfidf = TfidfVectorizer(stop_words='english')
    #Fit the vectorizer on the content column (fill missing values with empty string) and transform to a TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(df[content_col].fillna(""))
    #Create a mapping from item_id to the row index in the TF-IDF matrix, ensuring no duplicates and adding to the dictionary
    item_indices = pd.Series(df.index, index=df['item_id']).drop_duplicates().to_dict()

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

#Function to give recommendations to the user
def recommend_for_user(
    df,
    user_id,
    user_col='user_id',
    item_col='item_id',
    content_col='item_descrip',
    top_n=10,
    tfidf_matrix=None,
    item_indices=None
):
    """
    Recommend content-similar items based on what the user has interacted with
    """

    #Build the TF-IDF matrix and item index mapping from the content column
    tfidf_matrix, item_indices = build_content_matrix(df, content_col)
    #Get all unique items the user has interacted with
    user_items = df[df[user_col] == user_id][item_col].unique()

    #Dictionary to store the highest similarity score for each recommended item
    scores = {}
    for item_id in user_items:
        try:
            #Get similar items for each item the user has interacted with
            similar_items = get_similar_items(item_id, tfidf_matrix, item_indices, top_n=top_n)
            for sim_item, score in similar_items:
                #Only consider items the user hasn't already interacted with
                if sim_item not in user_items:
                    #Store the highest similarity score for each item
                    scores[sim_item] = max(scores.get(sim_item, 0), score)
        except KeyError:
            #Item not in index, skip to next
            continue  

    #Sort recommended items by similarity score in descending order and select top_n
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return ranked_items

#Function to print recommendations report
def print_recommendation_report(df, recommendations, item_col='item_id', content_col='item_descrip'):
    """
    Print recommended item details (item_id + description + score)
    """

    #Create a lookup dictionary from item_id to item_descrip
    id_to_desc = df.drop_duplicates(subset=[item_col]).set_index(item_col)[content_col].to_dict()

    #Print a header for the recommendations section
    print("\nTop Recommended Items:")
    #Iterate over each recommended item and its similarity score
    for item_id, score in recommendations:
        try:
            #Retrieve the description for the current item_id from the DataFrame
            description = df[df[item_col] == item_id][content_col].values[0]
        except IndexError:
            #If the item_id is not found, set description as "Unknown"
            description = "Unknown"
        #Print the item ID, similarity score, and description in a formatted string
        print(f"Item ID: {item_id:<5} | Similarity: {score:.4f} | Description: {description}")

#Function to show a report of what the user CHECKEDOUTed vs what was recommended
def show_recommendation_report_for_user(df, user_id, tfidf_matrix=None, item_indices=None, top_n=10):
    """
    Prints what the user CHECKOUTed vs. what the recommender returns (side-by-side).
    """
    # Print the user ID for context
    print(f"\n User ID: {user_id}")

    # Get CHECKOUTed items for the user (interaction_score == 3)
    checkout_df = df[(df['user_id'] == user_id) & (df['interaction_score'] == 3)][['item_id', 'item_descrip']].drop_duplicates()
    if checkout_df.empty:
        # Inform if the user has no CHECKOUT interactions
        print(" This user has no CHECKOUT interactions.")
        return

    # Print the list of items the user CHECKOUTed
    print("\n Items this user CHECKOUTed:")
    for _, row in checkout_df.iterrows():
        print(f"  - {row['item_id']: <5} | {row['item_descrip']}")

    # Get recommendations for the user using the content-based recommender
    from .content_based_fnb import recommend_for_user
    recs = recommend_for_user(df, user_id, top_n=top_n, tfidf_matrix=tfidf_matrix, item_indices=item_indices)

    if not recs:
        # Inform if no recommendations were returned
        print("\n No recommendations returned.")
        return

    # Print the top-N content-based recommendations
    print(f"\n Top {top_n} content-based recommendations:")
    # Create a lookup dictionary for item descriptions
    id_to_desc = df.drop_duplicates(subset=['item_id']).set_index('item_id')['item_descrip'].to_dict()

    # Print each recommended item with its description and similarity score
    for item_id, score in recs:
        desc = id_to_desc.get(item_id, "Unknown")
        print(f"  - {item_id: <5} | {desc} | similarity: {score:.4f}")
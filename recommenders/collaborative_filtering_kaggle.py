#imports
import numpy as np
from collections import defaultdict
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

#Function to normalize the Quantity column
def normalize_quantity(df, cap=None, log_scale=True):
    """
    Normalize the Quantity column.
    - cap: (int) max limit to clip the values
    - log_scale: (bool) apply log1p scaling
    """
    #Copy the DataFrame to avoid modifying the original
    df = df.copy()

    #If a cap value is provided, clip the Quantity values at the specified upper limit
    if cap:
        df['Quantity'] = df['Quantity'].clip(upper=cap)
    #If log_scale is True, apply log1p transformation to reduce skewness in Quantity values
    if log_scale:
        df['Quantity'] = df['Quantity'].apply(lambda x: np.log1p(x))
    return df

#Function for precision and recall at k
def precision_recall_at_k(predictions, k=10, threshold=1.0):
    """
    Compute Precision@K and Recall@K.
    """
    # Group predictions by user, storing (estimated_rating, true_rating) pairs for each user
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    # Initialize dictionaries to store precision and recall for each user
    precisions = {}
    recalls = {}

    # For each user, compute precision and recall at k
    for uid, user_ratings in user_est_true.items():
        # Sort the user's ratings by estimated rating in descending order
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Select the top k items for this user
        top_k = user_ratings[:k]

        # Count the number of relevant items (true rating >= threshold)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Count the number of recommended items in top k (estimated rating >= threshold)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        # Count the number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k)

        # Compute precision for this user
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        # Compute recall for this user
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    # Compute average precision and recall across all users
    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls)

    return avg_precision, avg_recall

#Function to train the user collaborative filtering model
def train_kaggle_user_cf(df, user_col='user_id', item_col='item_id', rating_col='Quantity'):
    """
    Train a user-based CF model using Kaggle dataset
    """
    #Create a Reader object with the rating scale from 1 to the maximum value in the rating column
    reader = Reader(rating_scale=(1, df[rating_col].max()))
    #Load the DataFrame into a Surprise Dataset using the specified user, item, and rating column
    data = Dataset.load_from_df(df[[user_col, item_col, rating_col]], reader)
    #Split the dataset into training and testing sets (80% train, 20% test)
    trainset, testset = train_test_split(data, test_size=0.2)

    #Initialise the KNNBasic algorithm for user-based collaborative filtering
    algo = KNNBasic(sim_options={'user_based': True})
    #Train the algorithm on the training set
    algo.fit(trainset)
    #Generate predictions on the test set
    predictions = algo.test(testset)
    #Print the Root Mean Squared Error (RMSE) of the predictions
    print("User-based CF RMSE (Kaggle):", accuracy.rmse(predictions))
    return algo

#Function to train the item collaborative filtering model
def train_kaggle_item_cf(df, user_col='user_id', item_col='item_id', rating_col='Quantity'):
    """
    Train an item-based CF model using Kaggle dataset
    """
    #Create a Reader object with the rating scale from 1 to the maximum value in the rating column
    reader = Reader(rating_scale=(1, df[rating_col].max()))
    #Load the DataFrame into a Surprise Dataset using the specified user, item, and rating column
    data = Dataset.load_from_df(df[[user_col, item_col, rating_col]], reader)
    #Split the dataset into training and testing sets (80% train, 20% test)
    trainset, testset = train_test_split(data, test_size=0.2)

    #Initialise the KNNBasic algorithm for item-based collaborative filtering
    algo = KNNBasic(sim_options={'user_based': False})
    #Train the algorithm on the training set
    algo.fit(trainset)
    #Generate predictions on the test set
    predictions = algo.test(testset)
    #Print the Root Mean Squared Error (RMSE) of the predictions
    print("Item-based CF RMSE (Kaggle):", accuracy.rmse(predictions))
    return algo
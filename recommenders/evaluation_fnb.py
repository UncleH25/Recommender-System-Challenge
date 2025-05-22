#imports
from collections import defaultdict

#Function to evaluate precision and recall
def evaluate_precision_recall(df, recommend_func, k=10, threshold_score=3, user_col='user_id', item_col='item_id'):
    """
    Evaluate average Precision@K and Recall@K for content-based recommendations
    Only evaluates users with at least 1 CHECKOUT item
    """

    #Select users who have at least one interaction at or above the threshold (e.g., CHECKOUT)
    users = df[df['interaction_score'] >= threshold_score][user_col].unique()
    #Lists to store precision and recall scores for each user
    precision_scores = []
    recall_scores = []

    #Iterate over each selected user
    for user_id in users:
        #Filter the DataFrame for the current user
        user_df = df[df[user_col] == user_id]

        #Ground truth: items this user CHECKOUTed (interaction_score >= threshold)
        true_items = set(user_df[user_df['interaction_score'] >= threshold_score][item_col])

        #Get recommendations for the user using the provided recommender function
        recs = recommend_func(df, user_id, top_n=k)
        #Extract the set of recommended item IDs
        recommended_items = set([item_id for item_id, _ in recs])

        #Skip users with no ground truth items
        if not true_items:
            continue

        #Calculate the intersection of recommended and true items (hits)
        hits = true_items & recommended_items
        #Compute precision: proportion of recommended items that are relevant
        precision = len(hits) / len(recommended_items) if recommended_items else 0
        #Compute recall: proportion of relevant items that are recommended
        recall = len(hits) / len(true_items) if true_items else 0

        #Store the precision and recall for this user
        precision_scores.append(precision)
        recall_scores.append(recall)

    #Calculate average precision and recall across all users
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    return avg_precision, avg_recall
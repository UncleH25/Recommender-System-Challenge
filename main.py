#imports
from analysis.kaggle_analysis import load_kaggle_data, analyse_kaggle_data
from analysis.fnb_analysis import load_fnb_data, analyse_fnb_data
import os

#main function
def main():
    #Prompt user for dataset choice
    print("=== Dataset Analysis ===")
    print("1. Analyse Kaggle dataset")
    print("2. Analyse FNB dataset")
    choice = input("Choose dataset to analyze (1 or 2): ")

    #If user chooses Kaggle dataset
    if choice == "1":
        kaggle_path = os.path.join("data", "ecommerce_data.csv")
        df = load_kaggle_data(kaggle_path)
        analyse_kaggle_data(df)
    #If user chooses FNB dataset
    elif choice == "2":
        fnb_path = os.path.join("data", "dq_ps_challenge_v2.csv")
        df = load_fnb_data(fnb_path)
        analyse_fnb_data(df)
    #If user chooses invalid option
    else:
        print("Invalid option.")

#Run the main function only if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()
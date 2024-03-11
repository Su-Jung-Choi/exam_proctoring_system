"""
Group Project (CSC746)
Project Title: Exam Proctoring System Using Face Detection
by Rajini Chittimalla, Sujung Choi, Madhu Sai Vineel Reka
File Description: This code is used to split the dataset into training and testing sets.
It divides the dataset into 80% training and 20% testing sets, and saves them to new csv files.
"""
import pandas as pd

files = ["normal.csv", "right.csv", "left.csv"]

for file in files:

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Split the original data into a training set (80%) and testing set (20%)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # save the training and testing data to new csv files
    train_df.to_csv(f'{file.replace(".csv", "_train.csv")}', index=False)
    test_df.to_csv(f'{file.replace(".csv", "_test.csv")}', index=False)
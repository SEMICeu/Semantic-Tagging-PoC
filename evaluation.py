
from semantic_tagging import read_pdf, read_word, predict_tags
import os
import pandas as pd

INPUT_PATH = ""

def comparing_to_ground_truth(relevant_tags, ground_truth):
    """"""
    number_of_tags = len(ground_truth)
    accuracy = 0

    for tag in relevant_tags:
        if tag in ground_truth:
            accuracy += 1

    return accuracy/number_of_tags

def main():

    # Loop through the test corpus 
    format_error = []

    testing_corpus = pd.read_excel(INPUT_PATH)
    accuracy_list = []
    prediction_list = []
    
    #for file in os.listdir(INPUT_PATH):
    for index, row in testing_corpus.iterrows():
        
        file = row[""]
        # Extracting the text from the file
        if file.endswith(".docx"):
            text = read_word(file)
        elif file.endswith(".pdf"):
            text = read_pdf(file)
        else:
            print("Wrong format")
            format_error.append(file)

        text = row[""]
        ground_truth = row[""]

        # Predicting tags on the extracted text
        predicted_tags = predict_tags(text)
        accuracy = comparing_to_ground_truth(predict_tags, ground_truth)

        prediction_list.append(predicted_tags)
        accuracy_list.append(accuracy)
    
    testing_corpus["Predictions"] = prediction_list
    testing_corpus["Accuracy"] = accuracy_list



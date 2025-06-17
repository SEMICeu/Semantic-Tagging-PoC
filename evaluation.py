
from semantic_tagging import predict_tags
import pandas as pd

INPUT_PATH = ""

def comparing_to_ground_truth(relevant_tags, ground_truth):
    """Series of operations needed to compute the accuracy of predicted tags"""
    number_of_tags = len(ground_truth)
    accuracy = 0

    for tag in relevant_tags:
        if tag in ground_truth:
            accuracy += 1

    return accuracy/number_of_tags, accuracy / len(relevant_tags)

def main():
    testing_corpus = pd.read_excel(INPUT_PATH)
    recall_list = []
    precision_list = []
    prediction_list = []

    # Loop through the test corpus 
    for index, row in testing_corpus.iterrows():

        text = row["Article summary "]
        ground_truth = row["DET tags that express the main subject matter (often with post-coordination)"].split(" + ")

        # Predicting tags on the extracted text
        predicted_tags = predict_tags(text)
        recall, precision = comparing_to_ground_truth(predicted_tags, ground_truth)

        prediction_list.append(predicted_tags)
        recall_list.append(recall)
        precision_list.append(precision)
    
    testing_corpus["Label created by Auto-tagger"] = prediction_list
    testing_corpus["Recall"] = recall_list
    testing_corpus["Precision"] = precision_list

    testing_corpus.to_excel("results/results.xlsx")

if __name__=="__main__":
    main()



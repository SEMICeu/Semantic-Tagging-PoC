
from semantic_tagging import read_pdf, read_word, predict_tags
import os
import pandas as pd
import PyPDF2
import requests
import io

INPUT_PATH = ""

def comparing_to_ground_truth(relevant_tags, ground_truth):
    """"""
    number_of_tags = len(ground_truth)
    accuracy = 0

    for tag in relevant_tags:
        if tag in ground_truth:
            accuracy += 1

    return accuracy/number_of_tags, accuracy / len(relevant_tags)

def main():

    # Loop through the test corpus 
    format_error = []

    testing_corpus = pd.read_excel(INPUT_PATH)
    recall_list = []
    precision_list = []
    prediction_list = []
    
    #for file in os.listdir(INPUT_PATH):
    for index, row in testing_corpus.iterrows():
        
        file = row["URL"]
        # Extracting the text from the file
        response = requests.get(file)
        if response.status_code == 200:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages: 
                text += page.extract_text() + "\n"
        else:
            print("Wrong format")
            format_error.append(file)

        text = row["Article summary "] # row["Article Title"]
        ground_truth = row["DET tags that express the main subject matter (often with post-coordination)"].split(" + ")

        # Predicting tags on the extracted text
        predicted_tags = predict_tags(text)
        recall, precision = comparing_to_ground_truth(predict_tags, ground_truth)

        prediction_list.append(predicted_tags)
        recall_list.append(recall)
        precision_list.append(precision)
    
    testing_corpus["Label created by Auto-tagger"] = prediction_list
    testing_corpus["Recall"] = recall_list
    testing_corpus["Precision"] = precision_list

    testing_corpus.to_excel("results/results.xlsx")

if __name__=="__main__":
    main()



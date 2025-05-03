
from semantic_tagging import read_pdf, read_word, predict_tags
import os

INPUT_PATH = ""

def comparing_to_ground_truth(relevant_tags, ground_truth):
    """TO DO"""

def main():

    # Loop through the test corpus 
    format_error = []
    for file in os.listdir(INPUT_PATH):
        
        # Extracting the text from the file
        if file.endswith(".docx"):
            text = read_word(file)
        elif file.endswith(".pdf"):
            text = read_pdf(file)
        else:
            print("Wrong format")
            format_error.append(file)

        # Predicting tags on the extracted text
        predicted_tags = predict_tags(text)
        comparing_to_ground_truth()



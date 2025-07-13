from semantic_tagging import predict_tags
import pandas as pd
import os

INPUT_PATH = "PATH_TO_YOUR_FILE"

def comparing_to_ground_truth(relevant_tags, ground_truth):
    """
    Compute the recall and precision of predicted tags against the ground truth tags.

    Recall is the ration of relevant tags (found in ground truth) over the total number
    of ground truth tags. Precision is the ration of relevant tags over the total number of 
    predicted tags.

    Args:
        relevant_tags(list): List of tags predicted by the model.
        ground_truth(list): List of actual relevant tags from the ground_truth.

    Returns:
        tuple: A tuple containing:
            - float: Recalll score
            - float: Precision score
    """
    number_of_tags = len(ground_truth)
    accuracy = sum(tag in ground_truth for tag in relevant_tags)

    # Calculate recall
    recall = accuracy / number_of_tags if number_of_tags > 0 else 0
    
    # Calculate precision
    precision = accuracy / len(relevant_tags) if len(relevant_tags) > 0 else 0

    return recall, precision

def main():
    """
    Main entry point of the script for evaluating the tag prediction.

    It reads the testing corpus from an Excel file, predicts tags for each article summary, 
    coomputes recall and precision, and saves the results in a new Excel file.
    """
    testing_corpus = pd.read_excel(INPUT_PATH)
    
    # Lists to store recall, precision, and predicted tags
    recall_list, precision_list, prediction_list = [], [], []

    # Loop through each row of the test corpus 
    for index, row in testing_corpus.iterrows():

        text = row["Article summary "]

        # Get ground truth tags and handle splitting and stripping
        ground_truth = row["DET tags that express the main subject matter (often with post-coordination)"].split(" + ")

        # Skip if ground truth is empty
        if not ground_truth:
            continue
       
        # Predicting tags using the provided model
        predicted_tags = predict_tags(text)

        # Compute recall and precision
        recall, precision = comparing_to_ground_truth(predicted_tags, ground_truth)

        # Append results to respective lists
        prediction_list.append(predicted_tags)
        recall_list.append(recall)
        precision_list.append(precision)
    
    # Add results to the DataFrame
    testing_corpus["Label created by Auto-tagger"] = prediction_list
    testing_corpus["Recall"] = recall_list
    testing_corpus["Precision"] = precision_list

    # Create ouotput directory if it does not exist
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)

    # Save the results to an Excel file
    output_file = os.path.join(output_dir, "results.xlsx")
    testing_corpus.to_excel(output_file, index=False)

    print(f'Results saved to {output_file}')

if __name__=="__main__":
    main()
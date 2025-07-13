import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import json
import pathlib
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchableField, 
    SearchField, 
    SimpleField, 
    SearchFieldDataType, 
    SemanticField, 
    VectorSearchProfile, 
    SemanticSettings, 
    VectorSearchAlgorithmKind, 
    HnswVectorSearchAlgorithmConfiguration, 
    SemanticConfiguration, 
    SearchIndex, 
    PrioritizedFields, 
    VectorSearch, 
    HnswParameters,
)

# Load pre-trained SentenceTransformer model
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
# Load environment variables from .env file
load_dotenv(override=True)

# Fetch Azure service endpoint, index name, and API key from environment variables
search_endpoint = os.environ.get("SEACRH_ENDPOINT")
index_name = os.environ.get("INDEX_NAME")
api_key = os.environ.get("API_KEY")
credential = AzureKeyCredential(api_key)

def creating_index_from_excel(file: pd.DataFrame, knowledge_base: list) -> list:
    """
    Creating a knowledge base from the provided Excel file.

    This functino extracts data from the DataFrame, encodes the definitions using
    a pre-trained model, and appends each document to the knowledge base list.

    Args:
        file (pd.DataFrame): The DataFrame containing the data loaded from Excel.
        knowledge_base (list): The list to which each document will be appended.

    Returns:
        list: The updated knowledge base list with generated documents.
    """
    id = 0
    for index, row in file.iterrows():
        # Only process rows where 'DEF' is not NaN
        if not f'{row["DEF"]}'=="nan": 
            document = {
                "id": f'{id}',
                "Label": (row["LIBELLE"] if not f'{row["LIBELLE"]}' == "nan" else ""),
                "Definition": (row["DEF"] if not f'{row["DEF"]}' == "nan" else ""), 
                "Label_def_vector": MODEL.encode(f'{row["LIBELLE"]}: {row["DEF"]}').tolist()
            }
            knowledge_base.append(document)
            id += 1


    return knowledge_base

def knowledge_base_to_json(knowledge_base):
    """
    Save each document in the knowledge base to a separate JSON file.

    This function writes each document in the knowledge base to a file in the 'eurovoc'
    directory with a sequential naming convention.

    Args:
        knowledge_base(list): The list of documents to be saved as JSON files.
    """
    path = os.getcwd()
    id = 1
    for doc in knowledge_base:
        id += 1
        file_path = os.path.join(path, 'eurovoc', f'eurovoc_{id}.json')

        # Write document data to JSON file
        with open(file_path, "w") as file: 
            json.dump(doc, file)


def create_or_update_index_on_Azure():
    """
    Create or update an Azure AI Search Index.

    This function defines the structure of the index, including fields and their types, 
    as well as vector search and semantic configurations, and submits to Azure.
    """
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

    # Define the fields for the index
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, retrievable=True),
        SearchableField(name="Label", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SearchableField(name="Definition", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SearchField(name="Label_def_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=384, vector_search_profile="myHnswProfile")

    ]

    # Configure vector search settings
    vector_search = VectorSearch(
        algorithms=[HnswVectorSearchAlgorithmConfiguration(name="myHnsw", kind=VectorSearchAlgorithmKind.HNSW, 
                                                           parameters=HnswParameters(m=4, ef_construction=400, ef_search=500, metric="cosine"))], 
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm="myHnsw")],
    )

    # Define semantic configuratino
    semantic_config = SemanticConfiguration(
        name="eurovoc-poc-semantic-tagging",
        prioritized_fields=PrioritizedFields(
            title_field=SemanticField(field_name="Label"),
            prioritized_content_fields=[SemanticField(field_name="Definition")],
            prioritized_keywords_fields=[SemanticField(field_name="Label")]

        )
    )

    # Combine semantic configurations
    semantic_settings = SemanticSettings(configurations=[semantic_config])

    # Create the index
    index = SearchIndex(
        name = index_name, fields = fields, vector_search=vector_search, semantic_settings=semantic_settings
    )

    # Submit the index creation or update the request to Azure
    result = index_client.create_or_update_index(index)
    print(f" {result.name} created")

def get_search_client():
    """
    Create and return a SearchClient instance for interacting with Azure AI search.

    Returns:
        SearchClient: An instance of SearchClient, configured with endpoint, index name, and credentials.
    """
    return SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)

def upload_index_to_Azure(folder_path: str): 
    """
    Upload JSON documents from a specified folder to Azure AI Search.

    This function loads JSON files from the given folder and uploads each document
    to the Azure search index. It logs any failures that occur during the upload process.

    Args:
        folder_path (str): The path to the folder containing the JSON files to upload.
    """
    index_client = get_search_client()
    index_entry_path = pathlib.Path(folder_path).glob("*")
    failed_to_upload = [] # List to keep track of failed upload attempts
    for entry_path in index_entry_path:
        print(entry_path)
        try:
            with open(entry_path) as file:
                index = json.load(file)
                print(f"Uploading {index['id']}")
                index_client.upload_documents(documents=[index])

        except Exception as e:
            print(entry_path)
            print(e)
            failed_to_upload.append(entry_path) # Record the path of the failed upload

    print(failed_to_upload)

def create_index():
    """
    Coordinate the creation of the Azure search index.

    This function reads an Excel file, prepares the knowledge base by converting it
    to JSON documents, creates or updates the Azure search index, and uploads the 
    documents to Azure.

    This is the main function that consolidates the steps necessary to set up
    the search infrastructure.
    """
    path = os.getcwd()
    
    index_folder = os.path.join(path, 'eurovoc')
    eurovoc_path = os.path.join(path, 'EuroVoc.xlsx')
    
    # Load Excel file into DataFrame
    eurovoc = pd.read_excel(eurovoc_path, sheet_name="desc_en")

    # Prepare knowledge base documents from Excel data
    print("DATA PREPARATION: create embedded_index")
    knowledge_base = []
    creating_index_from_excel(eurovoc, knowledge_base)
    knowledge_base_to_json(knowledge_base)

    # Create or update the index on Azure
    print("DATA PREPARATION: create_or_update_index_on_azure")
    create_or_update_index_on_Azure()

    # Upload the index documents to Azure
    print("DATA PREPARATION: upload_index_to_azure")
    upload_index_to_Azure(index_folder)

if __name__ == "__main__":
    create_index()
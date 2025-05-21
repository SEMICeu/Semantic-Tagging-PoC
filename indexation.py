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

MODEL = SentenceTransformer('all-MiniLM-L6-v2')
load_dotenv(override=True)

search_endpoint = os.environ.get("SEACRH_ENDPOINT")
index_name = os.environ.get("INDEX_NAME")
api_key = os.environ.get("API_KEY")
credential = AzureKeyCredential(api_key)

def creating_index_from_excel(file: pd.DataFrame, knowledge_base: list) -> list:
    id = 0
    for index, row in file.iterrows():
        
        if not f'{row["DEF"]}'=="nan": 
            document = {
                "id": f'{id}',
                "Label": (row["LIBELLE"] if not f'{row["LIBELLE"]}'=="nan" else ""),
                "Definition": (row["DEF"] if not f'{row["DEF"]}'=="nan" else ""), 
                "Label_def_vector": MODEL.encode(f'{row["LIBELLE"]}: {row["DEF"]}').tolist()
            }
            knowledge_base.append(document)
            id += 1


    return knowledge_base

def knowledge_base_to_json(knowledge_base):

    path = os.getcwd()
    id = 1
    for doc in knowledge_base:
        id += 1
        file_path = os.path.join(path, 'eurovoc', f'eurovoc_{id}.json')

        with open(file_path, "w") as file: 
            json.dump(doc, file)


def create_or_update_index_on_Azure():
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, retrievable=True),
        SearchableField(name="Label", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SearchableField(name="Definition", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SearchField(name="Label_def_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=384, vector_search_profile="myHnswProfile")

    ]

    vector_search = VectorSearch(
        algorithms=[HnswVectorSearchAlgorithmConfiguration(name="myHnsw", kind=VectorSearchAlgorithmKind.HNSW, 
                                                           parameters=HnswParameters(m=4, ef_construction=400, ef_search=500, metric="cosine"))], 
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm="myHnsw")],
    )

    semantic_config = SemanticConfiguration(
        name="eurovoc-poc-semantic-tagging",
        prioritized_fields=PrioritizedFields(
            title_field=SemanticField(field_name="Label"),
            prioritized_content_fields=[SemanticField(field_name="Definition")],
            prioritized_keywords_fields=[SemanticField(field_name="Label")]

        )
    )

    semantic_settings = SemanticSettings(configurations=[semantic_config])

    index = SearchIndex(
        name = index_name, fields = fields, vector_search=vector_search, semantic_settings=semantic_settings
    )

    result = index_client.create_or_update_index(index)
    print(f" {result.name} created")

def get_search_client():
    """Create and return a SearchClient instance"""
    return SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)

def upload_index_to_Azure(folder_path: str): 
    index_client = get_search_client()
    index_entry_path = pathlib.Path(folder_path).glob("*")
    failed_to_upload = []
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
            failed_to_upload.append(entry_path)

    print(failed_to_upload)

def create_index():
    path = os.getcwd()
    
    index_folder = os.path.join(path, 'eurovoc')
    eurovoc_path = os.path.join(path, 'EuroVoc.xlsx')
    
    eurovoc = pd.read_excel(eurovoc_path, sheet_name="desc_en")

    print("DATA PREPARATION: create embedded_index")
    knowledge_base = []
    creating_index_from_excel(eurovoc, knowledge_base)
    knowledge_base_to_json(knowledge_base)

    print("DATA PREPARATION: create_or_update_index_on_azure")
    create_or_update_index_on_Azure()
    print("DATA PREPARATION: upload_index_to_azure")
    upload_index_to_Azure(index_folder)

if __name__=="__main__":
    create_index()
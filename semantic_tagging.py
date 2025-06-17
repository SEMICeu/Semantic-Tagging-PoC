import os
from dotenv import load_dotenv
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import RawVectorQuery
import PyPDF2
#from docx import Document
import openai
from sentence_transformers import SentenceTransformer

# Set up Azure Search Client

load_dotenv(override=True)
search_endpoint = os.environ.get("SEACRH_ENDPOINT")
index_name = os.environ.get("INDEX_NAME")
api_key = os.environ.get("API_KEY")
print(search_endpoint)

azure_openai_api_key = os.environ.get("AZURE_OPENAI_KEY")
azure_openai_api_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
deployment_name = os.environ.get("DEPLOYMENT_NAME")
api_version = os.environ.get("API_VERSION")

credential = AzureKeyCredential(api_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)

model = SentenceTransformer('BAAI/bge-m3')

def read_pdf(file):
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = []
    for page in pdf_reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

def read_word(file):
    """Extract text from a word (.docx) file."""
    doc = ""#Document(file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)

def perform_search(query):
    """
    Perform semantic search on the Azure AI index and return top 10 tags.
    """
    try: 
        vector_query = RawVectorQuery(vector=model.encode(query).tolist(), k=50, fields="Label_def_vector")
        
        search_results = search_client.search(
            search_text=query, 
            vector_queries=[vector_query],
            top=50, 
        )
        
        tags = []
        for item in search_results:
            # Assuming that the search result has the field Label
            tags.append(item["Label"])
            if len(tags) >= 50:
                break
        return tags
    except Exception as e:
        st.error(f"An error occured: {e}")
        return []
    
from openai import AzureOpenAI
client = AzureOpenAI(api_key=azure_openai_api_key, azure_endpoint=azure_openai_api_endpoint, api_version=api_version)

from openai import AzureOpenAI
client = AzureOpenAI(api_key=azure_openai_api_key, azure_endpoint=azure_openai_api_endpoint, api_version=api_version)

def filter_with_LLM(user_input, search_results):
    """Use GPT-4 to filter the search results based on relevance"""
    prompt = (
        f"Based on the following document summary: '{user_input}', "
        f"evaluate the following EuroVoc descriptors and return only the relevant ones for annotating the document: {search_results}"
        f"Provide your answer as a list of maximum 10 relevant descriptors separated by commas."
    )

    response = client.chat.completions.create(
        model = deployment_name, 
        messages = [
            {"role": "system", "content": "Hello! You are a linguistic expert in charge of annotating documents with relevant tags from the EuroVoc thesaurus"},
            {"role": "user", "content": prompt}
        ], 
        max_tokens=150, 
        temperature=0
    )

    try:
        # Parse the response from the LLM
        print(response.choices[0])
        relevant_tags = response.choices[0].message.content.strip().split(',')
        relevant_tags = [tag.strip() for tag in relevant_tags if tag.strip()]
        return relevant_tags
    except Exception as e:
        st.error(f"Error processing with GPT-4: {e}")
        return []


def predict_tags(text):
    # Perform the search operation on the text
    tags = perform_search(text)

    # Use an LLM to filter the results
    relevant_tags = filter_with_LLM(text, tags)

    return relevant_tags


def main():
    # App title
    st.title("Semantic Tagging Solution")

    # Input Text Box
    user_text = st.text_area("Enter text for semantic tagging:")

    # File uploader for PDFs and Word documents
    uploaded_file = st.file_uploader("Or upload a PDF or Word document", type=["pdf", "docx"])

    # Button to trigger search
    if st.button("Generate tags"):
        if user_text.strip():
            text = user_text
        elif uploaded_file:
            if uploaded_file.type == "application/pdf":
                text = read_pdf(uploaded_file)
                print(text)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = read_word(uploaded_file)
            else: 
                st.error("Unsupported file format!")
                return
        else:
            st.error("Please enter some text or upload a file to analyse.")
            return
        
        # Perform the search operation on the text
        tags = perform_search(text)

        # Use an LLM to filter the results
        relevant_tags = filter_with_LLM(text, tags)

        # Display the tags
        if relevant_tags: 
            st.subheader("Generated Tags:")
            for tag in relevant_tags: 
                st.write(f"- {tag}")

if __name__ == "__main__":
    main()

import os
import re
import json
from dotenv import load_dotenv
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import RawVectorQuery
import PyPDF2
import re
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# Load environment variables from a .env file
load_dotenv(override=True)
search_endpoint = os.environ.get("SEACRH_ENDPOINT")
index_name = os.environ.get("INDEX_NAME")
api_key = os.environ.get("API_KEY")

azure_openai_api_key = os.environ.get("AZURE_OPENAI_KEY")
azure_openai_api_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
deployment_name = os.environ.get("DEPLOYMENT_NAME")
api_version = os.environ.get("API_VERSION")

# Initialise Azure Key Credential and Search Client
credential = AzureKeyCredential(api_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)

# Load the pre-trained SentenceTransformer model for text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_summary_with_gpt(text):
    """
    Generate a summary of the provided text using the GPT model.

    Args:
        text (str): The text to be summarised

    Returns:
        str: The generated summary
    """
    text = re.sub(r'\s+', ' ', text).strip() # Clean up whitespace

    max_length = 3000 * 4  # Set maximum length for the text

    # Truncate text if it exceeds the maximum length
    if len(text) > max_length:
        text = text[:max_length]

    # Create the prompt for the GPT model
    prompt = (
        f"You are a knowledgeable assistant that provides coherent and comprehensive summaries of text."
        f"Please summarize the following text in a clear and complete manner, ensuring that the summary "
        f"captures all essential points without cutting off in the middle of a sentence. Aim for about "
        f"3-5 sentences: \n\n{text}"
    )

    # Generate completion using GPT model
    response = client.chat.completions.create(
        model = deployment_name,
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes texts"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

def read_pdf(file):
    """
    Extract text from a PDF file.

    Args:
        file: The PDF file to extract text from.

    Returns:
        str: The extracted summary or generated summary if no summary found.
    """
    pdf_reader = PyPDF2.PdfReader(file)
    
    count = 0
    only_abstract = []

    summary = ""
    text = []

    # Iterate through each page in the PDF
    for page in pdf_reader.pages:
        page_content = page.extract_text()
        text.append(page_content)

        # Check for executive summary or abstract
        if "executive summary" in page_content.lower() or "abstract" in page_content.lower():
            if count > 0:
                summary = page_content
                if "executive summary" in page_content.lower() or only_abstract:
                    break # Stop collecting if we find a second summary
            if "executive summary" not in page_content.lower() and "abstract" in page_content.lower():
                only_abstract = True
            
            count += 1

    # If no summary is found, generate one using GPT
    if summary == "": 
        summary = generate_summary_with_gpt("\n".join(text))

    return summary


def perform_search(query):
    """
    Perform semantic search on the Azure index and return relevant tags.

    Args:
        query (str): The search query as a string

    Returns:
        list: A list of the top 10 relevant tags or an empty list if an error occurs.
    """
    try: 
        vector_query = RawVectorQuery(vector=model.encode(query).tolist(), k=10, fields="Label_def_vector")
        
        search_results = search_client.search(
            search_text=None, 
            vector_queries=[vector_query],
            top=10,
             
        )
        
        tags = []
        for item in search_results:
            # Assuming that the search result has the field Label
            tags.append(item["Label"])
            if len(tags) >= 10:
                break # Limit to the top 10 tags
        return tags
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []
    
# Initialise Azure OpenAI client for GPT-4
client = AzureOpenAI(api_key=azure_openai_api_key, azure_endpoint=azure_openai_api_endpoint, api_version=api_version)

def filter_with_LLM(user_input, search_results):
    """
    Use GPT-4 to filter the search results based on relevance.

    Args:
        user_input (str): The user's input to assess relevance.
        search_results (str): The list of tags returned from search.

    Returns:
        list: A list of relevant tags based on the user's input.
    """
    prompt = (
        f"Based on the following document summary: '{user_input}', "
        f"evaluate the following EuroVoc descriptors and return only the relevant ones for annotating the document: {search_results}."
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
        # Parse the response from the LLM to extract relevant tags
        print(response.choices[0])
        relevant_tags = response.choices[0].message.content.strip().split(',')
        relevant_tags = [tag.strip() for tag in relevant_tags if tag.strip()]
        return relevant_tags
    except Exception as e:
        st.error(f"Error processing with GPT-4: {e}")
        return []

def tags_with_LLM(user_input):
    """
    Propose EuroVoc descriptors for tagging a document using GPT-4.

    Args:
        user_input (str): A summary of the document content.

    Returns:
        list: A list of proposed Euro<voc descriptors.
    """
    prompt = (
        f"Can you propose EuroVoc descriptors for tagging this document with meaningful metadata about its content, based on its summary: '{user_input}', "
        f"Provide your answer as a list of relevant EuroVoc descriptors separated by commas. Avoid proposing descriptors about country or region names (ex: Turkye, Maghreb, ...)"
        )
    

    response = client.chat.completions.create(
        model = deployment_name, 
        messages = [
            {"role": "system", "content": "Hello! You are a linguistic expert in charge of annotating documents with relevant descriptors from the EuroVoc thesaurus"},
            {"role": "user", "content": prompt}
        ], 
        max_tokens=150, 
        temperature=0
    )

    try:
        # Parse the response from the LLM to extract relevant tags
        print(response.choices[0])
        relevant_tags = response.choices[0].message.content.strip().split(',')
        relevant_tags = [tag.strip() for tag in relevant_tags if tag.strip()]
        return relevant_tags
    except Exception as e:
        st.error(f"Error processing with GPT-4: {e}")
        return []

import json

# Load EuroVoc descriptors from a JSON file for validation
with open("EuroVoc.json") as json_file:
    EUROVOC = json.load(json_file)

def predict_tags(text):
    """
    Predict relevant tags for the provided text using a combination of searching and LLM filtering.

    Args:
        text (str): The input text for which tags are to be generated

    Returns:
        list: a list of relevant tags.
    """
    # Get tags using the LLM based on the user input
    tags = tags_with_LLM(user_input=text) 

    mapped_tags = []

    # Search for mappings to EuroVoc based on the initial tags obtained
    for tag in tags: 
        searched_tags = perform_search(tag)

        for item in searched_tags:
            mapped_tags.append(item)

    tags = tags + mapped_tags

    # Filter out non-EuroVoc descriptors
    tags = [item for item in tags if item in EUROVOC]
    # Drop duplicates
    tags = list(set(tags))

    # Use an LLM to filter the results
    relevant_tags = filter_with_LLM(text, tags)

    return relevant_tags

def refine_tags(user_input, tags, initial_text, chat_history):
    """
    Refine the list of EuroVoc descriptors based on the user feedback.

    Args:
        user_input (str): The user's comment for refining the tags.
        tags (list): The initial list of tags to be refined.
        initial_text (str): The summary of the document.
        chat_history (list): The history of previous interactions.

    Returns:
        list: The updated list of EuroVoc descriptors after refinement.
    """

    system_prompt = (
        f"You are an expert in the EuroVoc thesaurus, and you are here to help refine a list of EuroVoc descriptors used to describe a document. "
        f"The user will provide you with their comment about an initial list of descriptors, and you need to update this list based on these comments. "
        f"Provide your answer as an updated list of EuroVoc descriptors separated by commas based on the comment from the user. "
        f"The user has provided the summary of a document, along with EuroVoc descriptors (tags) that where generated for it. "
        f"**User Text:**{initial_text}"
        f"**Initial EuroVoc Tags:**{', '.join(tags)}"
    )

    prompt_text = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"The refinement history up until this point: \n{chat_history}"},
        {"role": "user", "content": f"The user's latest input:\n{user_input}"}
    ]

    response = client.chat.completions.create(
        model = deployment_name,
        messages = prompt_text,
        temperature=0,
        extra_body={
            "data_sources":[
                {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": search_endpoint,
                        "index_name": index_name,
                        "authentication": {
                            "type": "api_key",
                            "key": api_key,
                        }
                    }
                }
            ],
        }
    )

    try:
        # Parse the response from the LLM
        relevant_tags = response.choices[0].message.content.strip().split(',')
        relevant_tags = [tag.strip() for tag in relevant_tags if tag.strip()]
        return relevant_tags
    except Exception as e:
        st.error(f"Error processing with GPT-4: {e}")
        return []


@st.dialog("Ask for refinements")
def refine():
    """Streamlit dialog to handle user requests for tag refinement."""
    st.write("How can we improve the proposed list?")
    user_comment = st.text_input("Ask for refinements about the generated tags")
    if st.button("Run"):
        st.session_state.chat_history.append({"role": "user", "content": user_comment.strip()})
        refined_tags = refine_tags(
            user_comment,
            st.session_state.generated_tags,
            st.session_state.input_text,
            st.session_state.chat_history
        )
        st.session_state.generated_tags = refined_tags
        st.session_state.chat_history.append({"role": "assistant", "content": ", ".join(refined_tags)})
        st.rerun()

def main():
    """Main function to run the Streamlit application for semantic tagging."""
    st.title("Semantic Tagging Solution")

    if 'generated_tags' not in st.session_state:
        st.session_state.generated_tags = []
        st.session_state.input_text = ""
    if 'user_text' not in st.session_state:
        st.session_state.user_text = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if "show_refine" not in st.session_state:
        st.session_state.show_refine = False

    # Text area for user input
    user_text = st.text_area("Enter text for semantic tagging:", value=st.session_state.user_text)
    st.session_state.user_text = user_text

    # File uploader for PDF
    uploaded_file = st.file_uploader("Or upload a PDF or Word document", type=["pdf"])

    # Button to trigger tag generation
    if st.button("Generate tags"):
        if user_text.strip():
            text = user_text
        elif uploaded_file:
            if uploaded_file.type == "application/pdf":
                text = read_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = ""
            else: 
                st.error("Unsupported file format!")
                return
        else:
            st.error("Please enter some text or upload a file to analyse.")
            return

        # Update session state with parsed text and generated tags
        st.session_state.input_text = text
        st.session_state.generated_tags = predict_tags(text)

    # Display tags and refine button
    if st.session_state.generated_tags:
        cols = st.columns(3)
        for i, tag in enumerate(st.session_state.generated_tags):
            with cols[i % 3]:
                st.write(f"- {tag}")

        if st.button("Refine Tags"):
            st.session_state.show_refine = True

    # Show dialog box if triggered
    if st.session_state.show_refine:
        refine()
        st.session_state.show_refine = False
    
if __name__ == "__main__":
    main()


import os
from dotenv import load_dotenv
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import RawVectorQuery
import PyPDF2
import re
#from docx import Document
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

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_summary_with_gpt(text):
    text = re.sub(r'\s+', ' ', text).strip()

    max_length = 3000 * 4
    if len(text) > max_length:
        text = text[:max_length]

    prompt = (
        f"You are a knowledgeable assistant that provides coherent and comprehensive summaries of text."
        f"Please summarize the following text in a clear and complete manner, ensuring that the summary "
        f"captures all essential points without cutting off in the middle of a sentence. Aim for about "
        f"3-5 sentences: \n\n{text}"
    )

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
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    
    count = 0
    only_abstract = []

    summary = ""
    text = []

    for page in pdf_reader.pages:
        page_content = page.extract_text()
        text.append(page_content)

        # See if summary or abstract is provided in the PDF page
        if "executive summary" in page_content.lower() or "abstract" in page_content.lower():
            if count > 0:
                summary = page_content
                if "executive summary" in page_content.lower() or only_abstract:
                    break
            if "executive summary" not in page_content.lower() and "abstract" in page_content.lower():
                only_abstract = True
            
            count += 1

    if summary == "": 
        summary = generate_summary_with_gpt("\n".join(text))

    return summary
    

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
                break
        return tags
    except Exception as e:
        st.error(f"An error occured: {e}")
        return []
    
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

def tags_with_LLM(user_input):
    """Use GPT-4 to filter the search results based on relevance"""
    prompt = (
        f"Can you propose EuroVoc descriptors for tagging this document with meaningful metadata about its content, based on its summary: '{user_input}', "
        f"Provide your answer as a list of relevant EuroVoc descriptors separated by commas. Avoid proposing descriptors about country or region names (ex: Turkye, Maghreb, ...)")

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
        # Parse the response from the LLM
        print(response.choices[0])
        relevant_tags = response.choices[0].message.content.strip().split(',')
        relevant_tags = [tag.strip() for tag in relevant_tags if tag.strip()]
        return relevant_tags
    except Exception as e:
        st.error(f"Error processing with GPT-4: {e}")
        return []

import json

with open("EuroVoc.json") as json_file:
    EUROVOC = json.load(json_file)

def predict_tags(text):
    # Perform the search operation on the text
    tags = tags_with_LLM(user_input=text) # perform_search(text)

    mapped_tags = []

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

    system_prompt = (
        f"You are an expert in the EuroVoc thesaurus, and you are here to help refine a list of EuroVoc descriptors used to describe a document."
        f"The user will provide you with their comment about an initial list of descriptor, you need to update this list based on these comments."
        f"Provide your answer as an updated list of EuroVoc descriptors separated by commas based on the comment from the user. "
        f"The user has provided the summary of a document, along with EuroVoc descriptors (tags) that where generated for it."
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

    user_text = st.text_area("Enter text for semantic tagging:", value=st.session_state.user_text)
    st.session_state.user_text = user_text

    uploaded_file = st.file_uploader("Or upload a PDF or Word document", type=["pdf"])

    # Button to trigger tag generation
    if st.button("Generate tags"):
        if user_text.strip():
            text = user_text
        elif uploaded_file:
            if uploaded_file.type == "application/pdf":
                text = read_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = read_word(uploaded_file)
            else: 
                st.error("Unsupported file format!")
                return
        else:
            st.error("Please enter some text or upload a file to analyse.")
            return

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


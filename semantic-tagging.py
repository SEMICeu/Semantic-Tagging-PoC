import os
from dotenv import load_dotenv
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import PyPDF2
from docx import Document

# Set up Azure Search Client

load_dotenv()
search_endpoint = os.getenv("SEACRH_ENDPOINT")
index_name = os.getenv("INDEX_NAME")
api_key = os.getenv("API_KEY")

credential = AzureKeyCredential(api_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)

def read_pdf(file):
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = []
    for page in pdf_reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

def read_word(file):
    """Extract text from a word (.docx) file."""
    doc = Document(file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)

def perform_search(query):
    """
    Perform semantic search on the Azure AI index and return top 10 tags.
    """
    try: 
        search_results = search_client.search(query, include_total_count=True)
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

        # Display the tags
        if tags: 
            st.subheader("Generated Tags:")
            for tag in tags: 
                st.write(f"- {tag}")

if __name__ == "__main__":
    main()


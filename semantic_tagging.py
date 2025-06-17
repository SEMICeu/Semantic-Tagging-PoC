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

# Create functions for parsing input from user
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

# Functions for performing the semantic tagging
from openai import AzureOpenAI
client = AzureOpenAI(api_key=azure_openai_api_key, azure_endpoint=azure_openai_api_endpoint, api_version=api_version)

def tags_with_LLM(user_input):
    """Use GPT-4 to filter the search results based on relevance"""
    prompt = (
        f"Can you propose EuroVoc descriptors for tagging this document with meaningful metadata about its content, based on its summary: '{user_input}', "
        f"Provide your answer as a list of relevant EuroVoc descriptors separated by commas."
        f"Here are some examples you can take inspiration from:"
        f"Example 1:"
        f"[SUMMARY] This project has been inspired by the always more demanding need to upgrade the existing masonry EU buildings due to their poor seismic performance resulting in severe human and economic losses and their low energy performance which significantly increases their energy consumption. The issue of upgrading the unreinforced masonry (URM) buildings is of great importance since they are unengineered vernacular structures and far from the levels of the current standards for seismic capacity and energy consumption. Moreover, the latter combined with deterioration due to ageing of materials, environmental degradation, experience of several earthquakes and lack of maintenance, yield to even highers structural and energy deficiencies. Every recent moderate to high seismic shaking has caused damage ranging from cracks to partial or total collapse with a high death toll and economic loss. Therefore, this project aimed to confront both deficiencies in an integrated manner by exploring innovative techniques and advanced materials. The synergy of this project with the JRC institutional project iRESIST+ which focused on the upgrading of old RC buildings, tried to address the problem for the most common building typologies providing viable solutions in the context of EU regulations for energy savings of buildings and protection of cultural heritage."
        f"[OUTPUT] earthquake, disaster risk reduction"
        f"Example 2:"
        f"[SUMMARY] This report provides an empirical analysis of the drivers and barriers to adoption of autonomous machines (AM) technologies by European companies. Based on this analysis, the report provides a series of policy recommendations to accelerate the uptake of AM and robots and emphasise its impact on the European economy."
        f"[OUTPUT] smart technology, artificial intelligence, new technology, robotics, industrial policy"
        f"Example 3:"
        f"[SUMMARY] This years seminar was organized around a narrative which looked into the GAPS-PROCESS-SOLUTIONS-COMMUNICATION of scientific knowledge for DRM. Thanks to five panels of distinguished speakers and moderators, we touched upon all the above-mentioned topics: — The opening session confirmed the increasing role of the Science Pillar of the UCPKN as a space where existing and new scientific networks would be able to create, manage and share scientific knowledge and data with the aim of supporting decision makers in better anticipating, preparing for, and responding to disasters considering the changing landscape of the risks we are facing. — Session 1 and 2 drilled down on challenges of dealing with complex situations while orienting the research agendas and providing advice to policy makers — Session 3 focused on the DRM-climate change nexus, with specific attention to risk assessment methodologies for doing so, while informing –at the same time- the new processes of trans-boundary scenario building and of the Union Disaster Resilience Goals (UDRGs) — Session 4 made us reflect around the social dimension of DRM and specifically of the risk communication."
        f"[OUTPUT] scientific cooperation, disaster risk reduction, resilience"
        f"Example 4:"
        f"[SUMMARY] This report aims to explore potential concepts and architectures for the monitoring of the European Union’s disaster resilience goals. The report focuses on three main areas: (1) the use of the composite indicators approach for monitoring and review, (2) an exploration of potentially relevant indicators of resilience within the context of the disaster resilience goals and (3) demonstration of the Disaster Risk Management Knowledge Centre Risk Data Hub as a repository for the data reported and collected to facilitate its interpretation via maps and dashboards"
        f"[OUTPUT] civil defence, data collection, preparedness, disaster risk reduction, resilience"
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

    # Filter out non-EuroVoc descriptors
    tags = [item for item in tags if item in EUROVOC]
    # Drop duplicates
    tags = list(set(tags))

    return tags


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
        relevant_tags = predict_tags(text)

        # Display the tags
        if relevant_tags: 
            st.subheader("Generated Tags:")
            for tag in relevant_tags: 
                st.write(f"- {tag}")

if __name__ == "__main__":
    main()

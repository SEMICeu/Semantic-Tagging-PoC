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
from openai import AzureOpenAI, OpenAI
from qdrant_client import QdrantClient

# Load environment variables from a .env file
load_dotenv(override=True)
azure_deployment = os.environ.get("AZURE_DEPLOYMENT")

search_endpoint = os.environ.get("SEACRH_ENDPOINT")
if not search_endpoint:
    host = os.environ.get("HOST")
    port = os.environ.get("PORT")
index_name = os.environ.get("INDEX_NAME")
api_key = os.environ.get("API_KEY")

llm_api_key = os.environ.get("LLM_API_KEY")
llm_api_endpoint = os.environ.get("LLM_API_ENDPOINT")
deployment_name = os.environ.get("DEPLOYMENT_NAME")
api_version = os.environ.get("API_VERSION")

# Initialise Azure Key Credential and Search Client
if search_endpoint:
    credential = AzureKeyCredential(api_key)
    search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)
else: 
    search_client = QdrantClient(
            host=host, 
            port=port,
        )

# Initialise LLM client
if azure_deployment:
    client = AzureOpenAI(api_key=llm_api_key, azure_endpoint=llm_api_endpoint, api_version=api_version)
else: 
    client = OpenAI(api_key=llm_api_key, base_url=llm_api_endpoint)

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
f"""
You are an expert assistant trained to produce high‑quality summaries specifically designed to support EUR‑Lex semantic indexing.

Your task:
Provide a clear, complete, and coherent summary of the document text below, capturing all essential concepts needed for accurate subject indexing with EuroVoc.

------------------------------------------------------------
SUMMARY REQUIREMENTS
------------------------------------------------------------

1) Focus on **content, not metadata**
   - Summaries must describe what the document is ABOUT.
   - Do NOT emphasize non-subject metadata (type of act, form, procedure, publication info, authoring institution unless the content is ABOUT the institution).

2) Capture all **main ideas, substantive topics, issues, measures, actors, sectors, products, policies, legal areas**, and any **geographical scope** mentioned.
   - Summaries should retain all information that could influence EuroVoc descriptor selection.

3) Preserve **specificity**
   - If the text mentions detailed elements (e.g., product types, agreements, sectors, legal bases, state aid details, sanctions, financial mechanisms, case‑law issues), include them.

4) Maintain **neutral, factual, and concise wording**
   - No interpretation, judgment, or added context beyond what the text provides.
   - Rephrase faithfully without omitting essential concepts.

5) Ensure **coherence and completeness**
   - The summary must not cut off mid‑sentence.
   - It must form a complete narrative that covers all meaningful subject matter.

------------------------------------------------------------
INPUT TEXT
------------------------------------------------------------
{text}

------------------------------------------------------------
TASK
------------------------------------------------------------
Produce one paragraph (or more if needed) summarizing the document in a complete, precise, and concept‑rich manner suitable for downstream EuroVoc indexing.
"""
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
        
        if search_endpoint:
            search_results = search_client.search(
                search_text=None, 
                vector_queries=[vector_query],
                top=10,
                
            )
        else: 
            search_results = client.query_points(
                collection_name=index_name,
                query=[vector_query],
                limit=10,
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
f"""
You are an expert EUR‑Lex semantic indexer. 

Your task: Given (1) a document summary and (2) a list of candidate EuroVoc descriptors, select a maximum of 10 descriptors that best reflect the document’s content. 
You must follow the official EUR‑Lex indexing methodology exactly. 

------------------------------------------------------------ 
MANDATORY INDEXING RULES (from EUR‑Lex Indexing Policy) 
------------------------------------------------------------ 
1. Index ONLY the content of the document 
- Do NOT index physical or contextual metadata such as: 
    • type of document (regulation, directive, opinion, etc.) 
    • the institution as author (e.g. “European Commission”) unless the document is ABOUT that institution’s functioning or role 
    • applicant/defendant/parties or procedure type in case-law 
    • document form, publication type, or classification category 
- EuroVoc descriptors must represent the *subject matter*, not metadata. 
2. Apply the 3 stages of indexing 
    a. Understand the document content 
    b. Identify the main concepts with retrieval value c
    . Translate those concepts into EuroVoc descriptors 
3. Be as specific as possible 
    - Always choose the most specific descriptor available for a concept. 
    - Replace general descriptors with narrower ones when the narrow term fits. 
    Examples: 
        • use “trade agreement (EU)” instead of “agreement (EU)” 
        • use “citrus fruit” instead of “fruit” 
    - If no specific descriptor exists, use the closest broader descriptor. 
4. Do NOT use several descriptors from the same hierarchical line 
    - Never pair a broad term with its narrower term. 
    - Only choose the most specific one. 
    - Descriptors on the same hierarchy level (siblings) may be combined if both are relevant. 
5. Prefer pre‑coordinated descriptors 
    - Always choose EuroVoc descriptors that are explicitly EU‑contextualised when they exist: 
        • “import (EU)” instead of “import” 
        • “export (EU)” instead of “export” 
        • “financing of the EU budget” instead of generic “budget financing” 
        • “EU programme” instead of “action programme” 
    - Only combine simple descriptors when no pre‑coordinated descriptor covers the concept. 
6. Use geographical descriptors when appropriate 
    - Include a country or region descriptor when: 
        • the document explicitly concerns that country/region 
        • a procedure, aid measure, agreement or statistics relates to that country 
    - For groups of countries, prefer the group descriptor over listing each member. 
    - EU regions only (not regions of third countries). 
7. Avoid incorrect or out‑of‑context descriptors 
    - Do not infer content not present in the text. 
    - Ensure semantic alignment by checking contextual meaning, hierarchy, notes, USE/UF relations. 
    - A descriptor must reflect the actual document concepts, not suppositions. 
8. Combine descriptors only when necessary 
    - If no pre‑coordinated descriptor exists, combine simple descriptors to represent a compound concept. 
    - Ensure combinations do not violate hierarchical‑line constraints. 
9. Maintain consistency with EUR‑Lex indexing practice 
    - Use descriptors typically found for similar document types (agreements, state aid, trade, fisheries, market measures, case law, etc.). 
    - Prioritise descriptors that reflect: 
        • policy domain 
        • product type 
        • sector of activity 
        • measure/action (e.g. sanctions, aid, financing, approvals) 
        • geographical scope 
        • EU context (when relevant) 
        
------------------------------------------------------------ 
OUTPUT INSTRUCTIONS 
------------------------------------------------------------ 
Return **ONLY** a flat list of **maximum 10 EuroVoc descriptors**, separated by commas. 
Do not include explanations, numbers, bullets, or commentary. 
Do not add descriptors that are not in the candidate list. 

------------------------------------------------------------ 
INPUT 
------------------------------------------------------------ 

Document summary: {user_input} 
Candidate EuroVoc descriptors: {search_results} 

------------------------------------------------------------ 
TASK 
------------------------------------------------------------ 
Select the 10 (or fewer) descriptors from the candidate list that best satisfy the EUR‑Lex indexing rules above. 
Return the final list as comma-separated descriptors only.
"""
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
f"""
You are an expert EUR‑Lex semantic indexer.

Your task:
Given a document summary, propose a list of EuroVoc descriptors that best represent the document’s subject content (not metadata). Your output must be a comma-separated list of descriptors (maximum 10).

------------------------------------------------------------
MANDATORY EUR‑LEX INDEXING PRINCIPLES
------------------------------------------------------------

1) Index ONLY the content of the document
   - Do NOT index ‘physical entity’ metadata such as:
     • type/form of act (regulation, directive, opinion, etc.)
     • institution as author (e.g., “European Commission”) unless the content is ABOUT its role/competences/organisation
     • parties/applicant/defendant/procedure type (case law)
     • publication/classification categories
   - EuroVoc must represent subject matter: what the document is ABOUT.

2) Apply the 3 stages of indexing
   a. Understand the document (concepts and ideas)
   b. Identify principal concepts with retrieval value
   c. Express these concepts using EuroVoc descriptors (check domain, microthesaurus, USE/UF, NT/BT, RT, scope notes)

3) Be as specific as possible
   - Prefer the narrowest descriptor that correctly matches the concept.
   - Replace generic terms with specific ones when the specifics are present.
     Examples:
       • use “trade agreement (EU)” instead of “agreement (EU)”
       • use “citrus fruit” instead of “fruit”
   - If a specific descriptor does not exist, use the closest broader one.

4) Avoid hierarchical duplication
   - Do NOT select both a broad descriptor and its narrower term from the same hierarchical line.
   - Choose only the most specific applicable descriptor.
   - Sibling descriptors can be used together if both are relevant and not duplicative.

5) Prefer pre‑coordinated EU descriptors
   - Choose EU‑specific pre‑coordinated forms when available:
     • “import (EU)” rather than “import”
     • “export (EU)” rather than “export”
     • “EU programme” rather than “action programme”
     • “financing of the EU budget” rather than generic “budget financing”
   - Combine simple descriptors only if no suitable pre‑coordinated descriptor exists.

6) Use geographical descriptors when clearly relevant
   - Include a country/region when the document explicitly concerns that geography (e.g., state aid, agreement, statistics, origin of product).
   - For groups of countries or international organisations, prefer the group descriptor (e.g., “European Union”, “EEA”, “EFTA”) rather than listing all members.
   - Regions of EU Member States may be used; regions of third countries generally should not.

7) Avoid incorrect or out‑of‑context descriptors
   - Do not infer concepts not supported by the summary.
   - Ensure the descriptor’s semantic scope (notes, hierarchy, USE/UF) truly fits the document.

8) Combine descriptors only when necessary
   - If a compound concept lacks a single pre‑coordinated descriptor, use a minimal combination of simple descriptors—without violating rule (4) on hierarchical duplication.

9) Maintain consistency with EUR‑Lex practice
   - Favour descriptors that capture the substantive policy/measure/action, sector/product, EU context, and geography (when applicable).
   - Typical areas: agreements, state aid, trade/market measures, fisheries, budget/finance, market approval, sanctions, competition/mergers, case‑law subject matter, etc.

------------------------------------------------------------
OUTPUT REQUIREMENTS
------------------------------------------------------------
• Return ONLY a flat list of up to 10 EuroVoc descriptors, separated by commas.
• Do not include explanations, numbers, bullets, or commentary.
• Propose descriptors that exist in EuroVoc; prefer EU pre‑coordinated forms where applicable.

------------------------------------------------------------
INPUT
------------------------------------------------------------
Document summary:
{user_input}

------------------------------------------------------------
TASK
------------------------------------------------------------
Propose up to 10 EuroVoc descriptors that best represent the document content, following all rules above. Return a comma-separated list only.

"""
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


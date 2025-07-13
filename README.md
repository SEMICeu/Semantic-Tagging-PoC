Any comments on my Readme.md? # Semantic Tagging Solution

## Overview

**Semantic-Tagging-PoC** is a proof of concept resulting from a collaboration between DG DIGIT and the Publications Office of the EU (OP) on the use of Artificial Intelligence for semantic tagging.

The primary goal of this initiative was to explore various AI-based approaches for tagging documents with relevant semantic labels from a controlled vocabulary to enhance their searchability and organisation. The proof of concept specifically employed the EuroVoc multilingual thesaurus (an EU vocabulary of concepts) to tag text content. Different approaches were tested, but it was concluded that leveraging a combination of a search index and Generative AI was the most effective method for suggesting relevant EuroVoc descriptors for any given PDF document or text (see table below).

| Table with results of the evaluation |

Comprehensive details on the analysis of different methods can be found in the report available on the SEMIC support centre. Additionally, individuals interested in these various approaches can find their code available in dedicated branches within this repository.

As a proof of concept (PoC), this project was intended to test and validate different methodologies and showcase functionality, rather than serve as production-ready software. It provides a basic end-to-end pipeline from input text to suggested semantic tags, alongside a simple user interface for demonstration purposes.

## Features and Functionality

- **Automatic Semantic Tagging**: Given an input text or document, the tool automatically identifies and suggests relevant tags from the EuroVoc vocabulary. This aids in classifying the document's content by themes/topics.
- **Hybrid AI Approach**: The PoC combines semantic search with generative AI (GPT-4) to enhance tagging accuracy. It uses an Azure AI Search index of EuroVoc terms to identify initial candidate tags, and then an OpenAI GPT-4 model refines and filters these suggestions before presenting the final tags.
- **Support for Documents and Text**: The tool can process both free-text input and uploaded documents. It supports text input (e.g., users can paste a paragraph) and document upload (in PDF format) to extract text for tagging.
- **User Interface**: A simple web-based UI is provided for demonstration. Users can input text or upload a document, and the resulting tags are displayed in an easy-to-read format. Users can also interact with the tagging results; for example, after initial tags are generated, they can request refinements or provide feedback through the interface.
- **Configurable and Extendable**: Key parameters (such as the search service endpoint, index name, and AI model settings) are configurable in the code, allowing for adaptation to other semantic vocabularies or AI models as required. Various scripts are also available in the repository to index a new vocabulary or evaluate a new solution or approach. This flexibility facilitates experimentation with different data sources or tagging methods.

## Repository Structure

This repository contains the following main components:

- **semantic_tagging.py** – Core application script. This Python script contains the main logic for loading the AI models and search index, processing input text, and generating the tags. It also implements the web interface.
- **run_app.bat** – Launch script for Windows. A convenience script to set up the environment and run the application on Windows systems. Double-clicking this will install dependencies (if not already installed) and start the local web app.
- **run_app.sh** – Launch script for Mac/Linux. Similar to the above, but designed for Unix-based systems. It can be executed in a terminal to set up and launch the app on MacOS or Linux.
- **requirements.txt** – Python dependencies. This file lists the Python packages required to run the project. The installation scripts will utilise this file to install necessary libraries. (If no requirements file is present, the install script may handle dependencies automatically.)

In addition to the core application scripts, a few additional components are provided:

- **indexation.py** – Script for indexing the controlled vocabulary. This Python script contains the main logic for indexing the EuroVoc thesaurus (from EuroVoc.xlsx) into an Azure AI Search index.
- **evaluation.py** – Script for evaluating the PoC. This Python script contains the main logic for evaluating the PoC on a test corpus provided by the user. Results are stored in the `/results` folder.
- **README.md** – Project documentation (this file).

## Requirements

To run this application, the following is required:

- Python 3.8+
- Azure Subscription (with Azure Search and OpenAI resources)

## Technologies Used

- **Streamlit**: Framework for creating web apps for machine learning and data science.
- **Azure AI Search**: Service for implementing search functionalities.
- **OpenAI's GPT-4**: Advanced language model for generating summaries and filtering tags.
- **Sentence Transformers**: For encoding text to generate embeddings.

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/SEMICeu/Semantic-Tagging-PoC.git  
   cd Semantic-Tagging-PoC
   ```

2. Configuration:

Before running the application, you need to configure the environment variables. In the root directory, create a .env file and define the following variables:
```json
AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
AZURE_OPENAI_ENDPOINT=<your-azure-open-ai-endpoint>
DEPLOYMENT_NAME=<your-model-name>
API_VERSION=<your-azure-openai-version>

SEARCH_ENDPOINT<=your-azure-search-endpoint>
API_KEY=<your-azure-search-api-key>
INDEX_NAME=<your-azure-search-index-name>
```
See env.sample for an example.

3. Launch the app:

After configuration, you can start the PoC application. Use the appropriate launch script for your operating system:
- On Windows: Double-click `run_app.bat` or run it from a command prompt.
- On Linux/Mac: Open a terminal, navigate to the project folder, and execute `bash run_app.sh` (ensure it has execute permissions).

The first time you run the app, it will automatically install the required Python packages and dependencies. This setup may take a few minutes (approximately 5-10 minutes on the first run) as it downloads and installs the necessary libraries. You might be prompted to press a key or confirm during this process; follow any prompts in the console window.

Once the setup is complete, the script will launch the application. You should see a message indicating that a local server is running, and a web browser window should open automatically, pointing to the application’s interface. If a browser does not open by itself, check the console output for a local URL (for example, `http://localhost:5000`) and manually open that address in your web browser.

## Usage Guide

After launching, you can use the Semantic Tagging PoC through its simple web interface.

1. **Open the Application Interface**: Once the server is running, access the interface via your web browser (usually at `http://localhost` on a specified port, which the console will indicate). You should see a page with options to input text or upload a document.

2. **Input your Content**: You have two options:
   - **Option 1**: Paste/Type Text. There will be a text box where you can directly type or paste the text you want to tag. This is useful for short snippets, paragraphs, or any text content.
   - **Option 2**: Upload Document. You can choose a file from your computer (commonly a PDF) to upload. The tool will extract text from the file for processing. (Supported formats in this PoC are PDF)

3. **Generate Tags**: After providing the input (or uploading the file), click the "Generate Tags" button (the interface will have a clear call-to-action for this). The application will then process the text:
    - It sends the text to the Azure Cognitive Search index, retrieving a set of candidate EuroVoc terms that are semantically related to the content.
    - It then passes these candidates (along with context) to the GPT-4 model, which evaluates and filters them to produce a refined list of suggested tags.

   This process usually takes a few seconds to complete, depending on the length of the document and network latency.

4. **View Results**: The interface will display the recommended semantic tags for your document. Each tag corresponds to a EuroVoc concept that the system found relevant. For instance, if your text concerned renewable energy policies in Europe, you might see tags such as "Renewable energy", "Energy policy", "Sustainable development", etc., assuming those are EuroVoc terms relevant to the content. The tags provide a quick thematic overview of the document's subject matter.

5. **(Optional) Refine Tags**: The PoC interface allows you to refine the results. There may be an input field to "ask for refinements" or provide feedback. You can type a follow-up query or instruction if something is missing or if you seek different granularity. For instance, you could ask, "Only show more specific tags related to environmental policy," or "Why was 'Climate change' suggested?" – and the system (via the AI model) will adjust the tags or provide an explanation based on your request. This interactive loop is powered by the LLM, allowing you to iteratively improve the tagging output if needed.

6. **Repeat as Needed**: You can clear the input and try another text or document, or adjust your input and generate tags again. There’s no need to restart the app for each document; it can handle multiple uses in one session.

Output format: The tags are typically presented as a list of terms. In a full implementation, each tag could potentially be linked to a definition (since EuroVoc terms have definitions/IDs), but in this PoC they might just appear as plain text labels for simplicity. Use these tags as insights into the document's content or as metadata – for example, tags could be used to index the document in a content management system.

## Further Information

- **Documentation**: Currently, this README serves as the primary documentation. Additional documentation may be provided in future (for example, a wiki or GitHub Pages site) if the project is extended.
- **Related Projects**: This PoC is part of a series of “AI for Interoperability” experiments under SEMIC. You might also be interested in related prototypes, such as the GraphRAG PoC (Graph-based Retrieval-Augmented Generation chatbot for data modelling) and others. These are separate projects aiming to enhance semantic interoperability using AI.
- **SEMIC Community**: For context about the broader initiative, check out the Interoperable Europe portal’s section on SEMIC. There are blog posts, news, and the SEMIC annual conference materials showcasing how semantic interoperability solutions (like this PoC) are being used in government and public administration contexts.




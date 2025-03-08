**Aim: PDF-based AI chatbot with retrieval augmented generation**


**Architecture / Tech stack:**
 - Front-end: 
   - user interface via Gradio library
 - Back-end: 
   - HuggingFace embeddings
   - HuggingFace Inference API for open-source LLMs
   - Chromadb vector database
   - LangChain conversational retrieval chain


----

### Overview

**Description:**
This AI assistant, using Langchain and open-source LLMs, performs retrieval-augmented generation (RAG) from your PDF documents. The user interface explicitely shows multiple steps to help understand the RAG workflow. This chatbot takes past questions into account when generating answers (via conversational memory), and includes document references for clarity purposes. It leverages small LLM models to run directly on CPU hardware. 


### Local execution

Command line for execution:
> python app.py

The Gradio web application should now be accessible at http://localhost:7860


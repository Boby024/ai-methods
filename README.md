# AI-methods

<h2>Project 1: Movie Retrieval-Augmented Generation (RAG):</h2>
<span>
This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering natural language questions about movies using their plot descriptions (open to another column). It combines semantic search (via FAISS and sentence embeddings) with text generation (via a transformer model), providing accurate, context-aware responses based on a movie dataset.
</span><br>
<span>Stack Used:</span>
 <ul>
  <li>Dataset</li>
  <li>Embedding Model: sentence-transformers (e.g., all-MiniLM-L6-v2)</li>
  <li>Vector Store: FAISS</li>
  <li>Retriever: FAISS retriever</li>
  <li>RAG Pipeline: Custom</li>
</ul> 

<span>Steps:</span>
 <ul>
  <li>Load and preprocess movie data</li>
  <li>Embed movie descriptions (or plots)</li>
  <li>Index them with FAISS</li>
  <li>Query using a user question</li>
  <li>Retrieve top-k relevant contexts</li>
  <li>Generate an answer using a language model</li>
</ul>

<span>Extrac:</span>
 <ul>
  <li>Save and Load FAISS Index</li>
  <li>Logic to Use Stored Index</li>
</ul>

<br>
<h2>Project 2: Automated Document Analysis:</h2>
<span>
Many industries require document compliance checks, such as financial reports, legal contracts, or medical records. Traditionally, these checks require human review to verify text content and document structure (tables, forms, signatures, etc.). An AI-powered solution can streamline this process using LLMs and image processing.
</span>

Structural ideas for the first draft:
- Extract Text & Visual Elements
  - to extract printed or handwritten text from scanned documents
- Analyze Content with an LLM
  - Feed the extracted text to an LLM (such as BERT model) for semantic analysis, flagging compliance issues.
  - Use Named Entity Recognition (NER) to detect key terms, dates, and sensitive data

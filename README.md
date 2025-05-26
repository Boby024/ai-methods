# AI-methods

<h2>Movie Retrieval-Augmented Generation (RAG):</h2>
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
  <li>Save FAISS Index</li>
  <li>Load FAISS Index</li>
  <li>Logic to Use Stored Index</li>
</ul>


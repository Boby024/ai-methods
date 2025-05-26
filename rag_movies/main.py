import json
import os
import time
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


class DatasetLoader:
    def __init__(self, path, limit=100):
        self.path = path
        self.limit = limit
    
    def load(self)-> pd.DataFrame:
        df = pd.read_csv(self.path)
        if self.limit:
            df = pd.read_csv(self.path).head(self.limit)
        else:
            df = pd.read_csv(self.path)
        return df

    def preprocess(self):
        pass


class EmbedderBase:
    def __init__(self, data: pd.DataFrame, column: str=None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            data_path: Path to the movie (csv): from column
            model_name: Name of the sentence transformer model to use
        """
        self.df = data
        self.column = column
        self.model_name = model_name
        self.encoder = SentenceTransformer(self.model_name)

    def get_embeddings(self):
        print("Creating Embeddings for documents...")
        return self.encoder.encode(self.df[str(self.column)], convert_to_tensor=True).cpu().numpy()


class FaissRetriever:
    def __init__(self, embedder: EmbedderBase, index=None, index_path="faiss_index.index", index_meta_path="faiss_metadata.csv"):
        self.embedder = embedder
        self.index_path = index_path
        self.index_meta_path = index_meta_path
        self.index = None
    
    def _create_index(self):
        print("Creating FAISS index for documents ...")
        embeddings = self.embedder.get_embeddings()
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query_embedding, top_k=5):
        print("Retrieving result from the query ...")
        distances, indices = self.index.search(query_embedding, top_k)
        df_result = pd.DataFrame({'distances': distances[0], 'ann': indices[0]})
        df_merged = pd.merge(df_result, self.embedder.df, left_on='ann', right_index=True)
        return df_merged

    def save(self):
        print("Saving FAISS Index")
        metadata = self.embedder.df[str(self.embedder.column)]

        faiss.write_index(self.index, self.index_path)
        metadata.to_csv(self.index_meta_path, index=False)

    def load(self):
        print("Loading stored FAISS Index and metadata")
        self.index = faiss.read_index(self.index_path)

        df = DatasetLoader(self.index_meta_path).load()
        embedder = EmbedderBase(df, df.columns[0]) # there is only one column on the csv head file
        return FaissRetriever(embedder=embedder, index=self.index)


class RAGPipeline():
    def __init__(self, embedder: EmbedderBase, retriever: FaissRetriever, query: str, top_k=5, remove_faiss_index=False):
        self.retriever = retriever
        self.embedder = embedder
        self.query = query
        self.top_k = top_k
        self.remove_faiss_index = remove_faiss_index
    
    def remove_faiss_data(self, file_path):
        # Check if file exists before deleting
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} File deleted successfully.")
        else:
            print(f"{file_path} File does not exist.")
    
    def run(self):
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: User question
            k: Number of documents to retrieve
            
        Returns:
            DataFrame with relevant documents
        """
        start_time = time.time()

        if os.path.exists(self.retriever.index_path) and os.path.exists(self.retriever.index_meta_path):
            self.retriever.load()
        else:
            self.retriever._create_index()
            self.retriever.save()

        q_embed = self.embedder.encoder.encode([query.lower()])
        result = self.retriever.retrieve(q_embed, top_k=self.top_k)

        end_time = time.time()
        print(f"Running time: {end_time - start_time:.4f} seconds")

        return self.display_answer(result)

    def display_answer(self, result: pd.DataFrame = None):
        if result:
            for i, row in result.iterrows():
                pass



if __name__ == "__main__":
    df = DatasetLoader("data/wiki_movie_plots_deduped.csv", 1000).load()
    embedder = EmbedderBase(df, "Plot")
    retriever = FaissRetriever(embedder)

    while True:
        query = input("\nAsk a movie-related question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = RAGPipeline(embedder, retriever, query, 5).run()
        print("\nAnswer:", answer)

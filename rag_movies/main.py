import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import json


class DatasetLoader:
    def __init__(self, path, limit):
        self.path = path
        self.limit = limit
    
    def load(self):
        df = pd.read_csv(self.path)
        if self.limit:
            df = pd.read_csv(self.path).head(100)
        else:
            df = pd.read_csv(self.path)
        return df


class EmbedderBase:
    def __init__(self, data: pd.DataFrame, column: str, model_name: str = "all-MiniLM-L6-v2"):
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
    def __init__(self, embeddings, df: pd.DataFrame):
        self.embeddings = embeddings
        self.df = df
    
    def _create_index(self):
        print("Creating FAISS index for documents ...")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query_embedding, top_k=5):
        print("Retrieving result from the query ...")
        D, I = self.index.search(query_embedding, top_k)
        df_result = pd.DataFrame({'distances': D[0], 'ann': I[0]})
        df_merged = pd.merge(df_result, self.df, left_on='ann', right_index=True)
        return df_merged  


df = DatasetLoader("data/wiki_movie_plots_deduped.csv", 100).load()

embedder = EmbedderBase(df, "Plot")
encoder_model = embedder.encoder
embeddings = embedder.get_embeddings()

retriever = FaissRetriever(embeddings, df)
retriever._create_index()

query = "what movie talks about police investigations"
q_embed = encoder_model.encode([query])
contexts = retriever.retrieve(q_embed, top_k=3)

print(contexts)
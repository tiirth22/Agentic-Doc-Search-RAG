import re 
import os 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import Config

class VectorStoreManager : 

    def __init__(self):
        
        self.embeddings = HuggingFaceEmbeddings(model_name = Config.EMBEDDING_MODEL_NAME)
        
        self.vector_db_path = Config.VECTOR_DB_PATH

    
    def create_vector_store(self , chunks) : 
        """
        Creates a Chroma vector store from cleaned chunks.
        """
        print(f"--- Creating Vector Store at: {self.vector_db_path} ---")

        vector_db = Chroma.from_documents(
            documents=chunks ,
            embedding=self.embeddings , 
            persist_directory=self.vector_db_path
        )

        print("--- Vector Store created and persisted successfully. ---")
        return vector_db
    
    def get_retriever(self) : 
        """
        Loads the existing vector store and returns it as a retriever.
        """

        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(f"Vector store not found at {self.vector_db_path}. Please run ingestion first.")
        
        vector_db = Chroma(
            persist_directory=self.vector_db_path , 
            embedding_function=self.embeddings
        )

        retriever = vector_db.as_retriever(
            search_type = "similarity",
            search_kwargs={
                "k": Config.RETRIEVER_K
            }
        )

        return retriever
    

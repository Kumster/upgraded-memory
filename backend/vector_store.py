# backend/vector_store.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

class ComplaintVectorStore:
    """Vector store specifically for 311 complaint data"""
    
    def __init__(self, csv_path: str, index_path: str = "data/complaint_embeddings.index"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.csv_path = csv_path
        self.index_path = index_path
        self.df = None
        self.index = None
        
        # Load or create embeddings
        if os.path.exists(index_path) and os.path.exists(index_path + '.df.pkl'):
            self.load_index()
        else:
            self.create_index()
    
    def create_index(self):
        """Create embeddings from CSV data"""
        print("Creating complaint embeddings...")
        
        # Load data
        self.df = pd.read_csv(self.csv_path)
        
        # Create text descriptions for embedding
        # Combine complaint_type + descriptor + borough + location
        texts = []
        for _, row in self.df.iterrows():
            text = f"{row.get('complaint_type', '')} {row.get('descriptor', '')} " \
                   f"in {row.get('borough', '')} ZIP {row.get('incident_zip', '')}"
            texts.append(text)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} complaints...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        # Save index and dataframe
        self.save_index()
        print("✅ Complaint embeddings created and saved!")
    
    def save_index(self):
        """Save FAISS index and dataframe"""
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + '.df.pkl', 'wb') as f:
            pickle.dump(self.df, f)
    
    def load_index(self):
        """Load existing FAISS index and dataframe"""
        print("Loading existing complaint embeddings...")
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + '.df.pkl', 'rb') as f:
            self.df = pickle.load(f)
        print("✅ Complaint embeddings loaded!")
    
    def search(self, query: str, k: int = 5):
        """Search for relevant complaints"""
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        # Get results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            complaint = self.df.iloc[idx].to_dict()
            results.append({
                'complaint': complaint,
                'relevance_score': float(1 / (1 + dist)),
                'distance': float(dist)
            })
        
        return results
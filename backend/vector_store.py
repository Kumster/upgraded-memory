# backend/vector_store.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any

class Document:
    """Simple Document class for compatibility"""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class VectorStore:
    """
    Universal Vector Store that supports:
    1. NYC 311 Complaint Data (specialized for ComplaintAnalyzerAgent)
    2. General CSV data ingestion (for other agents)
    """
    
    def __init__(self, csv_path: str = "data/erm2-nwe9.csv"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.csv_path = csv_path
        self.index_path = "data/vector_store.index"
        self.df = None
        self.index = None
        self.documents = []
        
        # Try to load existing index
        if os.path.exists(self.index_path) and os.path.exists(self.index_path + '.data.pkl'):
            self.load_index()
        elif os.path.exists(csv_path):
            # Create index from CSV if it exists
            self.create_index_from_csv(csv_path)
        else:
            print(f"⚠️ CSV file not found at {csv_path}. Vector store is empty.")
            # Initialize empty index
            self.index = faiss.IndexFlatL2(384)  # 384 is the dimension for all-MiniLM-L6-v2
            self.df = pd.DataFrame()
    
    def create_index_from_csv(self, csv_path: str):
        """Create embeddings from CSV data (works for both complaint and general data)"""
        print(f"📚 Creating embeddings from {csv_path}...")
        
        try:
            # Load data
            self.df = pd.read_csv(csv_path)
            
            # Detect if this is complaint data or general data
            is_complaint_data = 'complaint_type' in self.df.columns
            
            # Create text descriptions for embedding
            texts = []
            metadata_list = []
            
            for idx, row in self.df.iterrows():
                if is_complaint_data:
                    # Specialized handling for NYC 311 complaint data
                    text = self._format_complaint_text(row)
                    metadata = {
                        'type': 'complaint',
                        'complaint_type': row.get('complaint_type', ''),
                        'descriptor': row.get('descriptor', ''),
                        'borough': row.get('borough', ''),
                        'zip': str(row.get('incident_zip', '')),
                        'created_date': str(row.get('created_date', '')),
                        'status': row.get('status', ''),
                        'row_index': idx
                    }
                else:
                    # General data handling - combine all columns
                    text = ' '.join([f"{col}: {row[col]}" for col in self.df.columns if pd.notna(row[col])])
                    metadata = {
                        'type': 'general',
                        'row_index': idx,
                        **{col: str(row[col]) for col in self.df.columns if pd.notna(row[col])}
                    }
                
                texts.append(text)
                metadata_list.append(metadata)
                
                # Create Document object
                self.documents.append(Document(page_content=text, metadata=metadata))
            
            # Generate embeddings
            print(f"🔢 Generating embeddings for {len(texts)} entries...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            # Save index
            self.save_index()
            print(f"✅ Vector store created with {len(texts)} entries!")
            
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            # Initialize empty index on error
            self.index = faiss.IndexFlatL2(384)
            self.df = pd.DataFrame()
            self.documents = []
    
    def _format_complaint_text(self, row: pd.Series) -> str:
        """Format complaint data for embedding"""
        parts = []
        
        if pd.notna(row.get('complaint_type')):
            parts.append(f"Complaint Type: {row['complaint_type']}")
        
        if pd.notna(row.get('descriptor')):
            parts.append(f"Description: {row['descriptor']}")
        
        if pd.notna(row.get('borough')):
            parts.append(f"Borough: {row['borough']}")
        
        if pd.notna(row.get('incident_zip')):
            parts.append(f"ZIP Code: {row['incident_zip']}")
        
        if pd.notna(row.get('incident_address')):
            parts.append(f"Address: {row['incident_address']}")
        
        if pd.notna(row.get('status')):
            parts.append(f"Status: {row['status']}")
        
        return '. '.join(parts)
    
    def save_index(self):
        """Save FAISS index and data"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + '.data.pkl', 'wb') as f:
            pickle.dump({
                'df': self.df,
                'documents': self.documents
            }, f)
    
    def load_index(self):
        """Load existing FAISS index and data"""
        print("📂 Loading existing vector store...")
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.index_path + '.data.pkl', 'rb') as f:
                data = pickle.load(f)
                self.df = data.get('df', pd.DataFrame())
                self.documents = data.get('documents', [])
            print(f"✅ Vector store loaded with {len(self.documents)} entries!")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            self.index = faiss.IndexFlatL2(384)
            self.df = pd.DataFrame()
            self.documents = []
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for relevant documents
        Returns: List of Document objects with page_content and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            print("⚠️ Vector store is empty. Returning empty results.")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Search
            distances, indices = self.index.search(
                query_embedding.astype('float32'), min(k, self.index.ntotal)
            )
            
            # Get results as Document objects
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    # Add relevance score to metadata
                    doc.metadata['relevance_score'] = float(1 / (1 + dist))
                    doc.metadata['distance'] = float(dist)
                    results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def search_complaints(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Specialized search for complaint data that returns dict format
        Used by ComplaintAnalyzerAgent
        """
        docs = self.search(query, k)
        
        results = []
        for doc in docs:
            if doc.metadata.get('type') == 'complaint':
                # Return complaint-specific format
                row_idx = doc.metadata.get('row_index', 0)
                if row_idx < len(self.df):
                    complaint = self.df.iloc[row_idx].to_dict()
                    results.append({
                        'complaint': complaint,
                        'relevance_score': doc.metadata.get('relevance_score', 0),
                        'distance': doc.metadata.get('distance', 0)
                    })
        
        return results
    
    def ingest_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ingest a new DataFrame into the vector store
        Useful for uploading custom CSV data
        """
        print(f"📥 Ingesting {len(df)} new rows...")
        
        try:
            # If vector store is empty, replace it
            if self.df is None or len(self.df) == 0:
                self.df = df
                self.create_index_from_csv(None)  # Create from current df
                return {
                    'status': 'success',
                    'rows_ingested': len(df),
                    'total_chunks': len(self.documents)
                }
            
            # Otherwise, append to existing data
            start_idx = len(self.df)
            self.df = pd.concat([self.df, df], ignore_index=True)
            
            # Create embeddings for new data
            texts = []
            for idx, row in df.iterrows():
                text = ' '.join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                texts.append(text)
                
                metadata = {
                    'type': 'general',
                    'row_index': start_idx + idx,
                    **{col: str(row[col]) for col in df.columns if pd.notna(row[col])}
                }
                self.documents.append(Document(page_content=text, metadata=metadata))
            
            # Generate new embeddings
            new_embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Add to index
            self.index.add(new_embeddings.astype('float32'))
            
            # Save updated index
            self.save_index()
            
            print(f"✅ Ingested {len(df)} rows. Total entries: {len(self.documents)}")
            
            return {
                'status': 'success',
                'rows_ingested': len(df),
                'total_chunks': len(self.documents)
            }
            
        except Exception as e:
            print(f"❌ Ingestion error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'rows_ingested': 0,
                'total_chunks': len(self.documents)
            }
    
    def get_collection_size(self) -> int:
        """Get the total number of documents in the vector store"""
        return len(self.documents)
    
    def get_complaints_by_borough(self, borough: str) -> List[Dict]:
        """Get all complaints for a specific borough"""
        if self.df is None or len(self.df) == 0:
            return []
        
        if 'borough' in self.df.columns:
            borough_df = self.df[self.df['borough'].str.upper() == borough.upper()]
            return borough_df.to_dict('records')
        
        return []
    
    def get_complaints_by_type(self, complaint_type: str) -> List[Dict]:
        """Get all complaints of a specific type"""
        if self.df is None or len(self.df) == 0:
            return []
        
        if 'complaint_type' in self.df.columns:
            type_df = self.df[self.df['complaint_type'].str.contains(complaint_type, case=False, na=False)]
            return type_df.to_dict('records')
        
        return []


# Backward compatibility aliases
ComplaintVectorStore = VectorStore
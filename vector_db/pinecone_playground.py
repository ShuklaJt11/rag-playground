import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

if __name__ == "__main__":
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        print("Successfully connected to Pinecone")
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Successfully loaded sentence transformer model")
        
        texts = [
            "Pinecone is a vector database.",
            "Vectors represent data in numerical form.",
            "Embedding models convert text to vectors."
        ]
        
        embeddings = model.encode(texts)
        print(f"Coverted {len(texts)} texts to embeddings")
        
        dimensions = embeddings.shape[1]
        print(f"Embedding dimension: {dimensions}")
        
        index_name = "rag-playground-index"
        
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=dimensions,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                )
            )
            print(f"Created Pinecone index '{index_name}'")
        else:
            print(f"Index '{index_name}' already exists.")
            
        index = pc.Index(index_name)
        print(f"Connected to index '{index_name}'")
        
        vector_data = [(str(i), embedding.tolist(), {"text": texts[i]}) for i, embedding in enumerate(embeddings)]
        print(f"Inserted {len(vector_data)} vectors into the index")
        
        query_text = "How does Pinecone work?"
        query_vector = model.encode([query_text])[0].tolist()
        print(f"Encoded query '{query_text}' into vector")
        
        results = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        
        print(results)
        
        if results.matches:
            print("\nQuery Results:")
            for match in results.matches:
                print(f"ID: {match.id}, Score: {match.score:.4f}\nText: {match.metadata['text']}\n")
        else:
            print("No matches found.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

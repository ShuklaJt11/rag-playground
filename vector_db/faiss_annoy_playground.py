import faiss
import numpy as np
import time

from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer

def faiss_search(embeddings, query_vector, k):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    distances, indices = index.search(np.array([query_vector]), k)
    return distances[0], indices[0]

def annoy_search(embeddings, query_vector, k):
    dimension = embeddings.shape[1]
    index = AnnoyIndex(dimension, "angular")
    
    for idx, embedding in enumerate(embeddings):
        index.add_item(idx, embedding)
    
    index.build(10)
    
    indices, distances = index.get_nns_by_vector(query_vector, k, include_distances=True)
    
    return distances, indices

if __name__ == "__main__":
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("SentenceTransformer model loaded.")
        
        texts = [
            "FAISS is a library for efficient similarity search.",
            "Vectors represent data in numerical form.",
            "Embedding models convert text to vectors.",
            "Local vector databases can be faster for small datasets.",
            "FAISS supports both CPU and GPU operations.",
            "Annoy is an efficient library for approximate nearest neighbor search.",
            "Vector databases are crucial for large-scale machine learning applications.",
            "Similarity search is used in recommendation systems and information retrieval.",
            "Dimensionality reduction techniques can improve search efficiency.",
            "Cosine similarity is a common metric in vector space models."
        ]
        
        embeddings = model.encode(texts)
        print(f"Converted {len(texts)} texts to embeddings.")
        
        dimension = embeddings.shape[1]
        print(f"Vector Dimension: {dimension}")
        
        query_text = input("Enter your query: ")
        query_vector = model.encode([query_text])[0]
        print(f"Encoded query: '{query_text}'")
        
        print("\nFaiss Results:\n")
        start_time = time.time()
        faiss_distances, faiss_indices = faiss_search(embeddings=embeddings, query_vector=query_vector, k=3)
        faiss_time = time.time() - start_time
        
        for idx, (distance, index) in enumerate(zip(faiss_distances, faiss_indices)):
            print(f"Rank {idx + 1}: Text: '{texts[index]}', Distance: {distance:.4f}, Score: {1 / (1 + distance):.4f}\n")
        print(f"FAISS Search Time: {faiss_time:.6f} seconds")
        
        print("\nAnnoy Results:\n")
        start_time = time.time()
        annoy_distances, annoy_indices = annoy_search(embeddings=embeddings, query_vector=query_vector, k=3)
        annoy_time = time.time() - start_time
        
        for idx, (distance, index) in enumerate(zip(annoy_distances, annoy_indices)):
            print(f"Rank {idx + 1}: Text: '{texts[index]}', Distance: {distance:.4f}, Score: {1 / (1 + distance):.4f}\n")
        print(f"ANNOY Search Time: {annoy_time:.6f} seconds")
        
    except Exception as e:
        print(f"An error occurred: {e}")       
        
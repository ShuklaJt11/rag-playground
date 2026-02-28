import librosa
import numpy as np
import torch
import torchvision.models as models

from gensim.models import Word2Vec
from networkx import karate_club_graph
from node2vec import Node2Vec
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


def word_embeddings():
    sentences = [["I", "love", "Manchester", "United"], ["I", "like", "F1"], ["I", "dispise", "Liverpool"]]
    model = Word2Vec(sentences, vector_size=15, window=5, min_count=1, workers=4)
    
    print(f"Embedding for 'love': {model.wv['love']}")
    print(f"Embedding for 'like': {model.wv['like']}")
    print(f"Embedding for 'dispise': {model.wv['dispise']}")
    

def sentence_embeddings():
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    
    sentences = ["I love Manchester United", "I like F1", "I dispise Liverpool"]
    embeddings = model.encode(sentences)
    
    print(f"Embedding for 'I love Manchester United': {embeddings[0]}")
    print(f"Embedding for 'I like F1': {embeddings[1]}")
    print(f"Sentence embedding shape: {embeddings.shape}")


def image_embeddings():
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[: -1]))
    model.eval()
    
    transfrom = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open("data/images/united_img.jpg")
    img_tensor = transfrom(img).unsqueeze(0)
    
    with torch.no_grad():
        embeddings = model(img_tensor).squeeze()
    
    print(f"Image Embedding shape: {embeddings.shape}")
    print(f"First few dimensions of image embeddings: {embeddings[:3]}")
    

def graph_embeddings():
    G = karate_club_graph()
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    print(f"Graph Embedding for node 0: {model.wv['0']}")
    

def audio_embeddings():
    file_path = "data/audio/beep-sound.mp3"
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    mfcc_embeddings = np.mean(mfccs.T, axis=0)
    
    print(f"Audio Embedding: {mfcc_embeddings}")
    
    
if __name__ == "__main__":
    print("=====================================================================")
    print("                            Word Embeddings")
    print("=====================================================================")
    word_embeddings()
    print("=====================================================================")
    print("                         Sentence Embeddings")
    print("=====================================================================")
    sentence_embeddings()
    print("=====================================================================")
    print("                           Image Embeddings")
    print("=====================================================================")
    image_embeddings()
    print("=====================================================================")
    print("                           Graph Embeddings")
    print("=====================================================================")
    graph_embeddings()
    print("=====================================================================")
    print("                           Audio Embeddings")
    print("=====================================================================")
    audio_embeddings()
    
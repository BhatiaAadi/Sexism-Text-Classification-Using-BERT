import pandas as pd
import numpy as np
import re
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Removing special characters
    3. Removing numbers
    4. Tokenizing
    5. Removing stopwords
    """
    # Convert to lowercase
    text = text.lower()
    
    
    # Remove user mentions (like [USER])
    text = re.sub(r'\[USER\]', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return ' '.join(tokens)

def build_tfidf_model(texts):
    """
    Build a TF-IDF vectorizer model using the provided texts
    """
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(texts)
    
    return vectorizer

def get_embedding(text, vectorizer):
    """
    Generate TF-IDF embeddings for a given text
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform text to TF-IDF vector
    vector = vectorizer.transform([processed_text])
    
    # Convert to dense representation
    dense_vector = vector.toarray()[0]
    
    return dense_vector

def generate_embeddings(df, vectorizer):
    """
    Generate embeddings for all texts in the dataframe
    """
    embeddings = []
    for text in df['text']:
        embedding = get_embedding(text, vectorizer)
        embeddings.append(embedding)
    
    return embeddings

def save_embeddings(embeddings, labels, output_file):
    """
    Save embeddings and labels as PyTorch tensors (.pt file)
    """
    # Convert embeddings list to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    # Create a dictionary to map label strings to indices
    label_to_idx = {}
    for label in labels.unique():
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)
    
    # Convert labels to tensor
    label_indices = [label_to_idx[label] for label in labels]
    labels_tensor = torch.tensor(label_indices, dtype=torch.long)
    
    # Save as dictionary
    data_dict = {
        'embeddings': embeddings_tensor,
        'labels': labels_tensor,
        'label_to_idx': label_to_idx
    }
    
    torch.save(data_dict, output_file)
    print(f"Embeddings saved to {output_file}")
    print(f"Embedding dimensions: {embeddings_tensor.shape}")
    print(f"Labels dimensions: {labels_tensor.shape}")
    print(f"Label mapping: {label_to_idx}")

def main():
    # Load dataset
    df = pd.read_csv('train.csv')
    
    # Preprocess all texts
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Build TF-IDF model
    vectorizer = build_tfidf_model(df['processed_text'])
    
    # Generate embeddings
    embeddings = generate_embeddings(df, vectorizer)
    
    # Save embeddings and labels
    save_embeddings(embeddings, df['label_vector'], 'tf-idf_embeddings.pt')
    
    # Sample usage of get_embedding function for new texts
    example_text = "This is a new example text to demonstrate the embedding function"
    new_embedding = get_embedding(example_text, vectorizer)
    print(f"Example embedding shape: {new_embedding.shape}")

if __name__ == "__main__":
    main()
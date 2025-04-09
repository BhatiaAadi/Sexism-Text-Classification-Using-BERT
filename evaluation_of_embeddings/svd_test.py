import pandas as pd
import numpy as np
import re
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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
    text = str(text).lower()
    
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

def generate_test_embeddings_cooccurrence(test_df, word_vectors_tensor, word_to_idx):
    """Generate embeddings for test data using pre-trained Co-occurrence+SVD model"""
    
    # Convert tensor to numpy for easier processing
    word_vectors = word_vectors_tensor.numpy()
    embedding_dim = word_vectors.shape[1]
    
    test_embeddings = []
    
    print(f"Generating Co-occurrence+SVD embeddings for {len(test_df)} test samples...")
    
    for i, text in enumerate(test_df['text']):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(test_df)}")
            
        # Preprocess text
        processed_text = preprocess_text(text)
        tokens = processed_text.split()
        
        # Get word vectors for each token in the text
        token_vectors = []
        for token in tokens:
            if token in word_to_idx:
                # Get the index of the token
                token_idx = word_to_idx[token]
                # Get the pre-computed word vector
                token_vectors.append(word_vectors[token_idx])
        
        # Calculate document embedding (average of word vectors)
        if token_vectors:
            doc_embedding = np.mean(token_vectors, axis=0)
        else:
            # If no tokens match, use zeros
            doc_embedding = np.zeros(embedding_dim)
            
        test_embeddings.append(doc_embedding)
    
    return test_embeddings

def save_embeddings(embeddings, labels, word_to_idx, word_vectors, output_file):
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
        'document_embeddings': embeddings_tensor,
        'labels': labels_tensor,
        'label_to_idx': label_to_idx,
        'word_to_idx': word_to_idx,
        'word_vectors': word_vectors
    }
    
    torch.save(data_dict, output_file)
    print(f"Embeddings saved to {output_file}")
    print(f"Embedding dimensions: {embeddings_tensor.shape}")
    print(f"Labels dimensions: {labels_tensor.shape}")

def main():
    # Load the test dataset
    print("Loading test dataset...")
    test_df = pd.read_csv('test.csv')
    
    # Load the pre-trained Co-occurrence+SVD model
    print("Loading pre-trained Co-occurrence+SVD model...")
    model_data = torch.load('cooccurrence_svd_embeddings.pt', map_location=torch.device('cpu'))
    
    # Extract necessary components
    word_vectors = model_data['word_vectors']
    word_to_idx = model_data['word_to_idx']
    
    print(f"Loaded Co-occurrence+SVD model with vocabulary size: {len(word_to_idx)}")
    print(f"Embedding dimension: {word_vectors.shape[1]}")
    
    # Generate embeddings for test data
    test_embeddings = generate_test_embeddings_cooccurrence(test_df, word_vectors, word_to_idx)
    
    # Save test embeddings
    save_embeddings(
        test_embeddings, 
        test_df['label_vector'], 
        word_to_idx, 
        word_vectors,
        'cooccurrence_svd_test_embeddings.pt'
    )
    
    print("Test embeddings generation complete!")

if __name__ == "__main__":
    main()
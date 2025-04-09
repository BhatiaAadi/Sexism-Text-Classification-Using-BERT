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

def generate_test_embeddings_word2vec(test_df, word_vectors, word_to_idx, embedding_dim):
    """Generate embeddings for test data using pre-trained Word2Vec model"""
    
    test_embeddings = []
    
    print(f"Generating Word2Vec embeddings for {len(test_df)} test samples...")
    
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
                # Get the word vector from the model
                token_vectors.append(word_vectors[token_idx].numpy())
        
        # Calculate document embedding (average of word vectors)
        if token_vectors:
            doc_embedding = np.mean(token_vectors, axis=0)
        else:
            # If no tokens match, use zeros
            doc_embedding = np.zeros(embedding_dim)
            
        test_embeddings.append(doc_embedding)
    
    return test_embeddings

def save_embeddings(embeddings, labels, word_to_idx, idx_to_word, word_vectors, embedding_dim, output_file):
    """
    Save embeddings and labels as PyTorch tensors (.pt file)
    """
    # Convert embeddings to tensor
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
        'idx_to_word': idx_to_word,
        'word_vectors': word_vectors,
        'embedding_type': 'word2vec_cbow_test',
        'embedding_dim': embedding_dim
    }
    
    torch.save(data_dict, output_file)
    print(f"Embeddings saved to {output_file}")
    print(f"Embedding dimensions: {embeddings_tensor.shape}")
    print(f"Labels dimensions: {labels_tensor.shape}")

def main():
    # Load the test dataset
    print("Loading test dataset...")
    test_df = pd.read_csv('test.csv')
    
    # Load the pre-trained Word2Vec model
    print("Loading pre-trained Word2Vec model...")
    model_data = torch.load('word2vec_cbow_embeddings.pt', map_location=torch.device('cpu'))
    
    # Extract necessary components
    word_vectors = model_data['word_vectors']
    word_to_idx = model_data['word_to_idx']
    idx_to_word = model_data['idx_to_word']
    embedding_dim = model_data['embedding_dim']
    
    print(f"Loaded Word2Vec model with vocabulary size: {len(word_to_idx)}")
    print(f"Embedding dimension: {embedding_dim}")
    
    # Generate embeddings for test data
    test_embeddings = generate_test_embeddings_word2vec(test_df, word_vectors, word_to_idx, embedding_dim)
    
    # Save test embeddings
    save_embeddings(
        test_embeddings, 
        test_df['label_vector'], 
        word_to_idx, 
        idx_to_word, 
        word_vectors,
        embedding_dim,
        'word2vec_test_embeddings.pt'
    )
    
    print("Test embeddings generation complete!")

if __name__ == "__main__":
    main()
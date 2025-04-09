import pandas as pd
import numpy as np
import re
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds
from collections import defaultdict, Counter

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
    
    return tokens

def build_vocabulary(all_tokens):
    """
    Build vocabulary from all tokens and create word-to-index mapping
    """
    # Count word frequencies
    word_counts = Counter([token for doc in all_tokens for token in doc])
    
    # Filter out rare words (optional)
    min_count = 1
    filtered_words = [word for word, count in word_counts.items() if count >= min_count]
    
    # Create word-to-index mapping
    word_to_idx = {word: idx for idx, word in enumerate(filtered_words)}
    
    return word_to_idx

def build_cooccurrence_matrix(all_tokens, word_to_idx, window_size=2):
    """
    Build co-occurrence matrix using a sliding window approach
    """
    vocab_size = len(word_to_idx)
    # Initialize sparse matrix
    cooc_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    
    # Iterate through each document
    for doc in all_tokens:
        # Consider only words in our vocabulary
        doc_indices = [word_to_idx[word] for word in doc if word in word_to_idx]
        
        # Iterate through each word and its context
        for i, center_word_idx in enumerate(doc_indices):
            # Context window: words to left and right of center word
            context_start = max(0, i - window_size)
            context_end = min(len(doc_indices), i + window_size + 1)
            
            # Update co-occurrence counts
            for j in range(context_start, context_end):
                if i != j:  # Don't count co-occurrence with self
                    context_word_idx = doc_indices[j]
                    # Can apply weighting based on distance if desired
                    distance = abs(i - j)
                    weight = 1.0 / distance if distance > 0 else 1.0
                    cooc_matrix[center_word_idx, context_word_idx] += weight
    
    # Convert to CSR format for efficient operations
    return cooc_matrix.tocsr()

def apply_svd(cooc_matrix, dim=300):
    """
    Apply SVD to the co-occurrence matrix and reduce dimensionality
    """
    print(f"Applying SVD to reduce dimensions to {dim}...")
    
    # Apply log to co-occurrence counts (optional, helps with very large counts)
    # Add 1 to avoid log(0)
    log_cooc = csr_matrix(np.log(cooc_matrix.toarray() + 1))
    
    # Apply SVD for dimensionality reduction
    # u contains word vectors
    u, s, vt = svds(log_cooc, k=dim)
    
    # Scale by singular values
    word_vectors = u * np.sqrt(s)
    
    return word_vectors

def get_embedding(text, word_to_idx, word_vectors):
    """
    Generate embedding for a given text by averaging word vectors
    """
    # Preprocess the text
    tokens = preprocess_text(text)
    
    # Get indices of tokens in vocabulary
    token_indices = [word_to_idx[token] for token in tokens if token in word_to_idx]
    
    if not token_indices:
        # If no tokens match vocabulary, return zero vector
        return np.zeros(word_vectors.shape[1])
    
    # Average word vectors
    embedding = np.mean([word_vectors[idx] for idx in token_indices], axis=0)
    
    return embedding

def generate_embeddings(df, word_to_idx, word_vectors):
    """
    Generate embeddings for all texts in the dataframe
    """
    embeddings = []
    for text in df['text']:
        embedding = get_embedding(text, word_to_idx, word_vectors)
        embeddings.append(embedding)
    
    return embeddings

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
    
    # Convert vocabulary and word vectors to tensors
    word_vectors_tensor = torch.tensor(word_vectors, dtype=torch.float32)
    
    # Save as dictionary
    data_dict = {
        'document_embeddings': embeddings_tensor,
        'labels': labels_tensor,
        'label_to_idx': label_to_idx,
        'word_to_idx': word_to_idx,
        'word_vectors': word_vectors_tensor
    }
    
    torch.save(data_dict, output_file)
    print(f"Embeddings saved to {output_file}")
    print(f"Document embedding dimensions: {embeddings_tensor.shape}")
    print(f"Word embedding dimensions: {word_vectors_tensor.shape}")
    print(f"Labels dimensions: {labels_tensor.shape}")
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Label mapping: {label_to_idx}")

def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('train.csv')
    
    # Preprocess all texts
    print("Preprocessing texts...")
    all_tokens = [preprocess_text(text) for text in df['text']]
    
    # Build vocabulary
    print("Building vocabulary...")
    word_to_idx = build_vocabulary(all_tokens)
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Build co-occurrence matrix
    print("Building co-occurrence matrix...")
    cooc_matrix = build_cooccurrence_matrix(all_tokens, word_to_idx, window_size=2)
    print(f"Co-occurrence matrix shape: {cooc_matrix.shape}")
    
    # Apply SVD for dimensionality reduction
    embedding_dim = 300  # Adjust as needed
    word_vectors = apply_svd(cooc_matrix, dim=embedding_dim)
    print(f"Word vectors shape: {word_vectors.shape}")
    
    # Generate document embeddings by averaging word vectors
    print("Generating document embeddings...")
    document_embeddings = generate_embeddings(df, word_to_idx, word_vectors)
    
    # Save embeddings and labels
    save_embeddings(document_embeddings, df['label_vector'], word_to_idx, word_vectors, 'cooccurrence_svd_embeddings.pt')
    
    # Example usage of get_embedding function for new texts
    example_text = "This is a new example text to demonstrate the embedding function"
    new_embedding = get_embedding(example_text, word_to_idx, word_vectors)
    print(f"Example embedding shape: {new_embedding.shape}")

if __name__ == "__main__":
    main()
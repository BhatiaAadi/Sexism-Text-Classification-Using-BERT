import pandas as pd
import numpy as np
import re
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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
    
    return data_dict

def load_embeddings(file_path):
    """
    Load embeddings from a .pt file
    """
    # Load with map_location to ensure CPU usage
    data = torch.load(file_path, map_location=torch.device('cpu'))
    
    # Extract embeddings based on the file structure
    if 'document_embeddings' in data:
        embeddings = data['document_embeddings']
    elif 'embeddings' in data:
        embeddings = data['embeddings']
    else:
        raise ValueError(f"Unknown embedding format in {file_path}")
    
    # Extract labels
    if 'labels' in data:
        labels = data['labels']
    else:
        raise ValueError(f"Labels not found in {file_path}")
    
    # Extract label mapping
    if 'label_to_idx' in data:
        label_to_idx = data['label_to_idx']
    else:
        label_to_idx = None
    
    return embeddings.numpy(), labels.numpy(), label_to_idx

def evaluate_embeddings(embeddings, labels):
    """
    Evaluate embeddings using silhouette score, nearest neighbor accuracy, and information gain
    """
    # Silhouette Score
    silhouette = silhouette_score(embeddings, labels)
    
    # Nearest Neighbor Accuracy
    knn = KNeighborsClassifier(n_neighbors=5)
    nn_accuracy = cross_val_score(knn, embeddings, labels, cv=5, scoring='accuracy').mean()
    
    # Information Gain / Mutual Information
    mi_scores = mutual_info_classif(embeddings, labels)
    avg_mi = np.mean(mi_scores)
    
    return {
        'silhouette_score': silhouette,
        'nn_accuracy': nn_accuracy,
        'avg_mutual_info': avg_mi
    }

def visualize_results(results):
    """
    Visualize the comparison results
    """
    df = pd.DataFrame(results).T
    
    # Create a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap='viridis', fmt='.4f')
    plt.title('Embedding Comparison Results')
    plt.savefig('embedding_comparison_results.png')
    plt.close()
    
    # Create a bar chart for each metric
    metrics = df.columns
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        df[metric].plot(kind='bar')
        plt.title(f'Comparison of {metric}')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f'{metric}_comparison.png')
        plt.close()
    
    return df

def main():
    print("Loading datasets...")
    # Load training data to build the TF-IDF model
    train_df = pd.read_csv('train.csv')
    
    # Load test data for evaluation
    test_df = pd.read_csv('test.csv')
    
    print("Preprocessing texts...")
    # Preprocess all texts
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    test_df['processed_text'] = test_df['text'].apply(preprocess_text)
    
    print("Building TF-IDF model...")
    # Build TF-IDF model on training data
    vectorizer = build_tfidf_model(train_df['processed_text'])
    
    print("Generating embeddings for test data...")
    # Generate embeddings for test data
    test_embeddings = generate_embeddings(test_df, vectorizer)
    
    print("Saving test embeddings...")
    # Save test embeddings and labels
    test_data_dict = save_embeddings(test_embeddings, test_df['label_vector'], 'tfidf_test_embeddings.pt')
    
    # Load other embedding methods if available
    embedding_results = {}
    embedding_files = {
        'TF-IDF': 'tfidf_test_embeddings.pt'
    }
    
    # Try to load Word2Vec and Co-occurrence+SVD embeddings if available
    other_embeddings = {
        'Word2Vec': 'word2vec_test_embeddings.pt',
        'Co-occurrence+SVD': 'cooccurrence_svd_test_embeddings.pt'
    }
    
    for name, file_path in other_embeddings.items():
        try:
            print(f"Attempting to load {name} embeddings from {file_path}...")
            embeddings, labels, _ = load_embeddings(file_path)
            embedding_files[name] = file_path
            print(f"Successfully loaded {name} embeddings with shape {embeddings.shape}")
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Skipping {name} embeddings.")
        except Exception as e:
            print(f"Error loading {name} embeddings: {e}")
    
    # Evaluate all available embeddings
    print("\nEvaluating embeddings...")
    for name, file_path in embedding_files.items():
        print(f"Evaluating {name} embeddings...")
        embeddings, labels, _ = load_embeddings(file_path)
        print(f"Shape: {embeddings.shape}, Labels: {labels.shape}")
        results = evaluate_embeddings(embeddings, labels)
        embedding_results[name] = results
        print(f"Results for {name}: {results}")
    
    # If we have multiple embedding methods, visualize the comparison
    if len(embedding_results) > 1:
        print("\nVisualizing embedding comparison...")
        results_df = visualize_results(embedding_results)
        print("\nEmbedding Comparison Results:")
        print(results_df)
        
        # Determine the best embedding for each metric
        for metric in results_df.columns:
            best_embedding = results_df[metric].idxmax()
            print(f"Best embedding for {metric}: {best_embedding} ({results_df[metric].max():.4f})")
    else:
        print("\nOnly one embedding method available. Skipping comparison visualization.")
        for name, results in embedding_results.items():
            print(f"Results for {name}:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
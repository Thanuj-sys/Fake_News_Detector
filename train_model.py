import os
try:
    import torch
except Exception:
    torch = None
import pandas as pd
from src.preprocess import preprocess_dataset, preprocess_text
from src.detector import FakeNewsDetector
import pickle

def train_and_save_model():
    print("Starting model training...")
    
    # Check for GPU
    if torch is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    else:
        print("PyTorch not available - using CPU-based models only")
    
    # Train on dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'fake_news_dataset.csv')
    if not os.path.exists(dataset_path):
        dataset_path = 'data/fake_news_dataset.csv'
    
    print(f"Loading dataset from {dataset_path}")
    df = preprocess_dataset(dataset_path)
    
    print("Initializing detector...")
    detector = FakeNewsDetector()
    
    print("Training detector (this may take a while)...")
    detector.train(df)
    
    # Save the trained model
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fake_news_detector.pkl')
    
    print(f"Saving trained model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(detector, f)
    
    print("Model training and saving complete!")
    return model_path

if __name__ == "__main__":
    model_path = train_and_save_model()
    print(f"Trained model saved to: {model_path}")
    print("You can now run web_ui.py to use the pre-trained model.")
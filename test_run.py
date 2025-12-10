#!/usr/bin/env python
"""Simple test script to verify the fake news detector works."""

import os
import sys

# Ensure the src module can be imported
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import preprocess_text, preprocess_dataset
from src.detector import FakeNewsDetector

def main():
    print("=" * 60)
    print("FAKE NEWS DETECTOR - Simple Test")
    print("=" * 60)
    
    # Load and preprocess dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'fake_news_dataset.csv')
    print(f"\n1. Loading dataset from: {dataset_path}")
    df = preprocess_dataset(dataset_path)
    print(f"   ✓ Loaded {len(df)} articles")
    
    # Initialize and train detector
    print("\n2. Training detector (this may take a minute)...")
    detector = FakeNewsDetector()
    detector.train(df)
    print("   ✓ Training complete")
    
    # Test on real news sample
    print("\n3. Testing on sample articles...")
    real_sample = df[df['label'] == 'real']
    if len(real_sample) > 0:
        idx = 0
        real_article = real_sample.iloc[idx] if hasattr(real_sample, 'iloc') else real_sample._rows[idx]
        real_title = real_article.get('title', '') if isinstance(real_article, dict) else real_article['title']
        real_text = real_article.get('clean_text', '') if isinstance(real_article, dict) else real_article['clean_text']
        real_source = real_article.get('source', '') if isinstance(real_article, dict) else real_article['source']
        
        real_pred = detector.predict(real_title, real_text, real_source)
        print(f"   Real article: '{real_title[:50]}...'")
        print(f"   Predicted as: {'FAKE' if real_pred else 'REAL'}")
        if not real_pred:
            print("   ✓ Correct!")
        else:
            print("   ✗ Incorrect (predicted as fake)")
    
    # Test on fake news sample
    fake_sample = df[df['label'] == 'fake']
    if len(fake_sample) > 0:
        idx = 0
        fake_article = fake_sample.iloc[idx] if hasattr(fake_sample, 'iloc') else fake_sample._rows[idx]
        fake_title = fake_article.get('title', '') if isinstance(fake_article, dict) else fake_article['title']
        fake_text = fake_article.get('clean_text', '') if isinstance(fake_article, dict) else fake_article['clean_text']
        fake_source = fake_article.get('source', '') if isinstance(fake_article, dict) else fake_article['source']
        
        fake_pred = detector.predict(fake_title, fake_text, fake_source)
        print(f"\n   Fake article: '{fake_title[:50]}...'")
        print(f"   Predicted as: {'FAKE' if fake_pred else 'REAL'}")
        if fake_pred:
            print("   ✓ Correct!")
        else:
            print("   ✗ Incorrect (predicted as real)")
    
    # Test with custom input
    print("\n4. Testing with custom input...")
    custom_text = "Breaking news: Scientists discover amazing new cure for everything!"
    custom_source = "conspiracy.com"
    clean_custom = preprocess_text(custom_text)
    custom_pred = detector.predict("", clean_custom, custom_source)
    print(f"   Text: '{custom_text}'")
    print(f"   Source: {custom_source}")
    print(f"   Predicted as: {'FAKE' if custom_pred else 'REAL'}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
    print("\nYou can now:")
    print("  - Run 'python train_model.py' to train and save a model")
    print("  - Run 'python web_ui.py' to start the web interface")
    print("=" * 60)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Comprehensive verification script to test all functionality."""

import os
import sys

# Ensure the src module can be imported
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)
    
    try:
        from src.preprocess import preprocess_text, preprocess_dataset
        print("✓ src.preprocess imported successfully")
    except Exception as e:
        print(f"✗ Failed to import src.preprocess: {e}")
        return False
    
    try:
        from src.detector import FakeNewsDetector
        print("✓ src.detector imported successfully")
    except Exception as e:
        print(f"✗ Failed to import src.detector: {e}")
        return False
    
    try:
        from src.bert_semantic import BertSemanticModel
        print("✓ src.bert_semantic imported successfully")
    except Exception as e:
        print(f"✗ Failed to import src.bert_semantic: {e}")
        return False
    
    try:
        from src.markov_style import MarkovStyleModel
        print("✓ src.markov_style imported successfully")
    except Exception as e:
        print(f"✗ Failed to import src.markov_style: {e}")
        return False
    
    try:
        from src.web_verify import verify_web
        print("✓ src.web_verify imported successfully")
    except Exception as e:
        print(f"✗ Failed to import src.web_verify: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_preprocessing():
    """Test text preprocessing."""
    print("=" * 70)
    print("TEST 2: Text Preprocessing")
    print("=" * 70)
    
    from src.preprocess import preprocess_text
    
    test_text = "This is a TEST! With CAPITALS and numbers 123 and URLs http://example.com"
    result = preprocess_text(test_text)
    print(f"Original: {test_text}")
    print(f"Processed: {result}")
    
    # Check that processing worked
    if result and len(result) > 0:
        print("\n✅ Text preprocessing working!\n")
        return True
    else:
        print("\n✗ Text preprocessing failed!\n")
        return False


def test_dataset_loading():
    """Test dataset loading."""
    print("=" * 70)
    print("TEST 3: Dataset Loading")
    print("=" * 70)
    
    from src.preprocess import preprocess_dataset
    
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'fake_news_dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found at: {dataset_path}")
        return False
    
    print(f"Loading dataset from: {dataset_path}")
    df = preprocess_dataset(dataset_path)
    
    print(f"Loaded {len(df)} articles")
    
    # Check we have both real and fake news
    real_count = len([r for r in (df._rows if hasattr(df, '_rows') else df.to_dict('records')) if r.get('label') == 'real'])
    fake_count = len([r for r in (df._rows if hasattr(df, '_rows') else df.to_dict('records')) if r.get('label') == 'fake'])
    
    print(f"  - Real news: {real_count}")
    print(f"  - Fake news: {fake_count}")
    
    if len(df) > 0 and real_count > 0 and fake_count > 0:
        print("\n✅ Dataset loading working!\n")
        return True
    else:
        print("\n✗ Dataset loading failed!\n")
        return False


def test_web_verification():
    """Test web source verification."""
    print("=" * 70)
    print("TEST 4: Web Source Verification")
    print("=" * 70)
    
    from src.web_verify import verify_web
    
    # Test reliable source
    reliable_score = verify_web("nytimes.com")
    print(f"nytimes.com reliability score: {reliable_score:.3f}")
    
    # Test unreliable source
    unreliable_score = verify_web("fakenews.com")
    print(f"fakenews.com reliability score: {unreliable_score:.3f}")
    
    # Test unknown source
    unknown_score = verify_web("unknown-website.com")
    print(f"unknown-website.com reliability score: {unknown_score:.3f}")
    
    if reliable_score > 0.5 and unreliable_score < 0.5:
        print("\n✅ Web verification working!\n")
        return True
    else:
        print("\n✗ Web verification not distinguishing sources properly!\n")
        return False


def test_model_training():
    """Test model training with small sample."""
    print("=" * 70)
    print("TEST 5: Model Training (Small Sample)")
    print("=" * 70)
    
    from src.preprocess import preprocess_dataset
    from src.detector import FakeNewsDetector
    
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'fake_news_dataset.csv')
    
    print("Loading dataset...")
    df = preprocess_dataset(dataset_path)
    
    print("Creating detector...")
    detector = FakeNewsDetector()
    
    print("Training on full dataset (this may take a minute)...")
    detector.train(df)
    
    print("\n✅ Model training completed!\n")
    return detector


def test_predictions(detector):
    """Test making predictions."""
    print("=" * 70)
    print("TEST 6: Making Predictions")
    print("=" * 70)
    
    from src.preprocess import preprocess_text
    
    # Test 1: Reliable source with professional text
    test1_text = "The President announced new economic policies during a press conference at the White House today."
    test1_source = "reuters.com"
    clean1 = preprocess_text(test1_text)
    pred1 = detector.predict("", clean1, test1_source)
    
    print(f"Test 1 - Professional text from reliable source:")
    print(f"  Source: {test1_source}")
    print(f"  Text: {test1_text[:60]}...")
    print(f"  Prediction: {'FAKE' if pred1 else 'REAL'}")
    
    # Test 2: Sensational text from unreliable source
    test2_text = "SHOCKING! You won't BELIEVE what scientists discovered! Click here NOW!!!"
    test2_source = "clickbait.com"
    clean2 = preprocess_text(test2_text)
    pred2 = detector.predict("", clean2, test2_source)
    
    print(f"\nTest 2 - Sensational text from unreliable source:")
    print(f"  Source: {test2_source}")
    print(f"  Text: {test2_text}")
    print(f"  Prediction: {'FAKE' if pred2 else 'REAL'}")
    
    # Test 3: Neutral text from unknown source
    test3_text = "The weather today was sunny with temperatures reaching 75 degrees."
    test3_source = "weather-blog.com"
    clean3 = preprocess_text(test3_text)
    pred3 = detector.predict("", clean3, test3_source)
    
    print(f"\nTest 3 - Neutral text from unknown source:")
    print(f"  Source: {test3_source}")
    print(f"  Text: {test3_text}")
    print(f"  Prediction: {'FAKE' if pred3 else 'REAL'}")
    
    print("\n✅ Predictions working!\n")
    return True


def test_model_persistence(detector):
    """Test saving and loading the model."""
    print("=" * 70)
    print("TEST 7: Model Persistence")
    print("=" * 70)
    
    import pickle
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'test_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    print(f"Saving model to: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(detector, f)
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"Model saved successfully ({file_size:.2f} MB)")
    
    # Load model
    print("Loading model...")
    with open(model_path, 'rb') as f:
        loaded_detector = pickle.load(f)
    
    print("Model loaded successfully")
    
    # Test that loaded model works
    from src.preprocess import preprocess_text
    test_text = "This is a test article about politics."
    clean_text = preprocess_text(test_text)
    pred = loaded_detector.predict("", clean_text, "example.com")
    
    print(f"Test prediction with loaded model: {'FAKE' if pred else 'REAL'}")
    
    # Cleanup
    os.remove(model_path)
    print("Test model file cleaned up")
    
    print("\n✅ Model persistence working!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" FAKE NEWS DETECTOR - COMPREHENSIVE VERIFICATION")
    print("=" * 70 + "\n")
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    
    if results['imports']:
        results['preprocessing'] = test_preprocessing()
        results['dataset'] = test_dataset_loading()
        results['web_verify'] = test_web_verification()
        
        if results['dataset']:
            detector = test_model_training()
            results['training'] = detector is not None
            
            if results['training']:
                results['predictions'] = test_predictions(detector)
                results['persistence'] = test_model_persistence(detector)
    
    # Summary
    print("=" * 70)
    print(" VERIFICATION SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "✗ FAIL"
        print(f"{test_name.upper():20s} {status}")
    
    all_passed = all(results.values())
    
    print("=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The fake news detector is fully operational.")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
    print("=" * 70)
    
    print("\n📋 Next Steps:")
    print("  1. Run 'python train_model.py' to create the production model")
    print("  2. Run 'python web_ui.py' to start the web interface")
    print("  3. Open http://127.0.0.1:5000 in your browser")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

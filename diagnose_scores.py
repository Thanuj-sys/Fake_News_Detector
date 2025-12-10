#!/usr/bin/env python3
"""Diagnose why predictions are the same"""

import sys
sys.path.insert(0, r'd:\probaliti\fake_news_detector\fake_news_detector')

from src.detector import FakeNewsDetector
from src.preprocess import preprocess_text
import pickle

# Load model
model_path = r'd:\probaliti\fake_news_detector\fake_news_detector\models\fake_news_detector.pkl'
with open(model_path, 'rb') as f:
    detector = pickle.load(f)

print('\n' + '='*80)
print(' DIAGNOSTIC: Component Scores Breakdown')
print('='*80)

# Test cases
tests = [
    ("Reuters - REAL", "The Federal Reserve announced today that it will maintain current interest rates.", "reuters.com", "REAL"),
    ("Clickbait - FAKE", "SHOCKING!!! You will not BELIEVE this AMAZING discovery! Click NOW!!!", "clickbait.fake", "FAKE"),
]

for name, text, source, expected in tests:
    print(f'\n[{name}] Expected: {expected}')
    print('-'*80)
    
    clean_text = preprocess_text(text)
    
    # Get individual scores (matching detector.py logic)
    real_score = detector.markov_real.style_score(clean_text)
    fake_score = detector.markov_fake.style_score(clean_text)
    import math
    style_prob = 1 / (1 + math.exp(real_score - fake_score))
    
    # TF-IDF similarity
    vec = detector.tfidf_vectorizer.transform([clean_text]).toarray()
    from sklearn.metrics.pairwise import cosine_similarity
    real_sim = cosine_similarity(vec, detector.tfidf_real_centroid.reshape(1, -1))[0,0]
    fake_sim = cosine_similarity(vec, detector.tfidf_fake_centroid.reshape(1, -1))[0,0]
    denom = real_sim + fake_sim
    tfidf_prob = float(fake_sim / denom) if denom > 0 else 0.5
    
    bert_prob = detector.bert_model.predict(clean_text)
    
    from src.web_verify import verify_web
    web_score = verify_web(source)
    
    # Calculate combined (same formula as in detector.py)
    combined_prob = (
        0.35 * style_prob +
        0.35 * tfidf_prob +
        0.2 * bert_prob +
        0.1 * (1 - web_score)
    )
    
    prediction = "FAKE" if combined_prob > 0.5 else "REAL"
    
    print(f'Text: {text[:60]}...')
    print(f'Source: {source}')
    print(f'\nComponent Scores:')
    print(f'  1. Markov Style:  {style_prob:.4f} (weight: 35%)')
    print(f'     Real score: {real_score:.4f}, Fake score: {fake_score:.4f}')
    print(f'  2. TF-IDF:        {tfidf_prob:.4f} (weight: 35%)')
    print(f'     Real sim: {real_sim:.4f}, Fake sim: {fake_sim:.4f}')
    print(f'  3. BERT:          {bert_prob:.4f} (weight: 20%)')
    print(f'  4. Web Source:    {web_score:.4f} -> inverted: {(1-web_score):.4f} (weight: 10%)')
    print(f'\nWeighted Contributions:')
    print(f'  Style:   {0.35 * style_prob:.4f}')
    print(f'  TF-IDF:  {0.35 * tfidf_prob:.4f}')
    print(f'  BERT:    {0.2 * bert_prob:.4f}')
    print(f'  Web:     {0.1 * (1-web_score):.4f}')
    print(f'  ----------------')
    print(f'  COMBINED: {combined_prob:.4f}')
    print(f'\nPrediction: {prediction} (threshold: 0.5)')
    print(f'Result: {"✅ CORRECT" if prediction == expected else "❌ WRONG"}')

print('\n' + '='*80)
print(' DIAGNOSIS')
print('='*80)

# Check if all scores are identical
print('\n🔍 Checking for issues:')
print('   - If all component scores are ~0.5, they are not distinguishing')
print('   - Style and TF-IDF should vary between real and fake articles')
print('   - Web source should be different for reuters.com vs clickbait.fake')
print('='*80 + '\n')

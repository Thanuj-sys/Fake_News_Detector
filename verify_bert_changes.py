#!/usr/bin/env python3
"""Direct test to verify BERT scores are changing"""

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
print(' BERT SCORE VERIFICATION - Testing if scores change')
print('='*80)

# Check if BERT is stub or real
bert_is_stub = getattr(detector.bert_model, 'stub', False)
print(f'\nBERT Model Type: {"STUB (Random)" if bert_is_stub else "REAL (Transformers)"}')

# Test multiple different texts
tests = [
    ("Real News 1", "The Federal Reserve announced interest rates remain stable.", "REAL"),
    ("Real News 2", "Scientists published research in Nature journal today.", "REAL"),
    ("Real News 3", "The company reported quarterly earnings exceeding expectations.", "REAL"),
    ("Fake News 1", "SHOCKING!!! You won't BELIEVE this discovery!!!", "FAKE"),
    ("Fake News 2", "Click NOW before they DELETE this truth!!!", "FAKE"),
    ("Fake News 3", "Wake up sheeple! The government is HIDING everything!", "FAKE"),
]

print('\n' + '-'*80)
print(' BERT Predictions for Different Articles')
print('-'*80)

bert_scores = []
for name, text, expected in tests:
    clean_text = preprocess_text(text)
    bert_prob = detector.bert_model.predict(clean_text)
    bert_scores.append(bert_prob)
    
    print(f'{name:15} | Text: {text[:45]:45} | BERT: {bert_prob:.4f}')

print('-'*80)

# Check if scores are identical
unique_scores = len(set(round(s, 4) for s in bert_scores))
print(f'\nUnique BERT scores: {unique_scores} out of {len(bert_scores)} tests')

if unique_scores == 1:
    print('❌ PROBLEM: All BERT scores are IDENTICAL!')
    print(f'   All returning: {bert_scores[0]:.4f}')
    print('\n   This means BERT is NOT responding to different inputs.')
    print('   The model may not be loaded correctly or is still using stub.')
elif unique_scores <= 2:
    print('⚠️  WARNING: Very few unique scores - limited variance')
    print(f'   Unique values: {set(round(s, 4) for s in bert_scores)}')
else:
    print('✅ SUCCESS: BERT scores are varying!')
    print(f'   Range: {min(bert_scores):.4f} to {max(bert_scores):.4f}')
    print(f'   Variance: {max(bert_scores) - min(bert_scores):.4f}')

# Test full predictions
print('\n' + '='*80)
print(' FULL DETECTOR PREDICTIONS (All Components)')
print('='*80)

test_pairs = [
    ("Reuters Real", "The Federal Reserve maintains current rates.", "reuters.com", "REAL"),
    ("Clickbait Fake", "SHOCKING!!! Click NOW before DELETED!!!", "clickbait.fake", "FAKE"),
]

for name, text, source, expected in test_pairs:
    clean_text = preprocess_text(text)
    is_fake = detector.predict('', clean_text, source)
    result = "FAKE" if is_fake else "REAL"
    status = "✅" if result == expected else "❌"
    
    print(f'\n{status} {name}:')
    print(f'   Expected: {expected} | Got: {result}')

print('\n' + '='*80 + '\n')

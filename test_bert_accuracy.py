#!/usr/bin/env python3
"""Test BERT model accuracy independently"""

import sys
sys.path.insert(0, r'd:\probaliti\fake_news_detector\fake_news_detector')

from src.bert_semantic import BertSemanticModel
from src.preprocess import preprocess_text
import pickle

print('\n' + '='*80)
print(' BERT MODEL ACCURACY TEST')
print('='*80)

# Load the trained detector to get the BERT model
model_path = r'd:\probaliti\fake_news_detector\fake_news_detector\models\fake_news_detector.pkl'
with open(model_path, 'rb') as f:
    detector = pickle.load(f)

bert_model = detector.bert_model

# Check if using stub or real BERT
print(f'\nBERT Model Type: {"STUB (LogisticRegression)" if getattr(bert_model, "stub", False) else "REAL (Transformers)"}')
if hasattr(bert_model, 'hidden_size'):
    print(f'Hidden Size: {bert_model.hidden_size}')

print('\n' + '='*80)
print(' Testing BERT Predictions on Various Articles')
print('='*80)

# Test cases: (text, expected_label, description)
test_cases = [
    (
        "The Federal Reserve announced today that it will maintain current interest rates following a two-day policy meeting in Washington.",
        "REAL",
        "Reuters - Professional news"
    ),
    (
        "SHOCKING!!! You will not BELIEVE this AMAZING discovery! Click NOW before this gets DELETED!!!",
        "FAKE",
        "Clickbait - Sensational"
    ),
    (
        "Researchers at Oxford University published findings in the journal Nature examining climate change effects.",
        "REAL",
        "BBC - Science news"
    ),
    (
        "Anonymous sources reveal SHOCKING conspiracy by global elites! Wake up sheeple!",
        "FAKE",
        "Conspiracy theory"
    ),
    (
        "Apple Inc. confirmed today that its latest iPhone model will be available for pre-order starting next Friday.",
        "REAL",
        "Tech news - TechCrunch"
    ),
    (
        "Big Pharma is trying to HIDE this from you! Share this with EVERYONE before it's too late!",
        "FAKE",
        "Conspiracy/Medical misinformation"
    ),
    (
        "The company reported quarterly earnings that exceeded analyst expectations by a significant margin.",
        "REAL",
        "Business news"
    ),
    (
        "You won't believe what happens next!!! This ONE weird trick will change your life FOREVER!!!",
        "FAKE",
        "Clickbait headline"
    ),
]

correct = 0
total = len(test_cases)

print('\nRunning tests...\n')

for i, (text, expected, description) in enumerate(test_cases, 1):
    clean_text = preprocess_text(text)
    prob_fake = bert_model.predict(clean_text)
    
    # BERT returns probability of being fake (0.0 to 1.0)
    predicted = "FAKE" if prob_fake > 0.5 else "REAL"
    is_correct = (predicted == expected)
    
    if is_correct:
        correct += 1
    
    status = "✅" if is_correct else "❌"
    print(f'{status} Test {i}: {description}')
    print(f'   Text: {text[:60]}...')
    print(f'   Expected: {expected} | Predicted: {predicted} | Prob(Fake): {prob_fake:.3f}')
    print()

print('='*80)
print(f' BERT MODEL ACCURACY: {correct}/{total} ({int(correct/total*100)}%)')
print('='*80)

# Additional info
print('\n📊 Analysis:')
if getattr(bert_model, 'stub', False):
    print('⚠️  Currently using STUB model (LogisticRegression fallback)')
    print('   - The stub uses random embeddings, so accuracy may be low')
    print('   - To use real BERT, install: pip install transformers torch')
    print('   - Real BERT would provide better semantic understanding')
else:
    print('✅ Using real BERT model (transformers library)')
    print(f'   - Accuracy: {int(correct/total*100)}%')
    if correct/total >= 0.7:
        print('   - Performance is good for semantic analysis')
    elif correct/total >= 0.5:
        print('   - Performance is moderate - may need more training')
    else:
        print('   - Performance is low - model needs retraining or better data')

print('\n💡 Note: BERT is just ONE component (20% weight) in the ensemble')
print('   - Markov Style: 35% weight')
print('   - TF-IDF: 35% weight')
print('   - BERT: 20% weight')
print('   - Web Source: 10% weight')
print('   The overall detector combines all signals for final prediction.\n')

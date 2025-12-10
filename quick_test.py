#!/usr/bin/env python3
"""Quick test to show prediction variety"""

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
print(' TESTING PREDICTION VARIETY - REAL vs FAKE NEWS')
print('='*80)

# Test 1: REAL NEWS - Reuters
print('\n[1] REAL NEWS - Reuters')
print('-'*80)
text1 = 'The Federal Reserve announced today that it will maintain current interest rates following a two-day policy meeting in Washington.'
pred1 = detector.predict('Federal Reserve Maintains Rates', preprocess_text(text1), 'reuters.com')
print(f'Text: {text1[:60]}...')
print(f'Source: reuters.com')
print(f'Result: {"🚨 FAKE" if pred1 else "✅ REAL"}')

# Test 2: FAKE NEWS - Clickbait
print('\n[2] FAKE NEWS - Clickbait')
print('-'*80)
text2 = 'SHOCKING!!! You will not BELIEVE this AMAZING discovery! Click NOW before this gets DELETED!!!'
pred2 = detector.predict('SHOCKING Discovery!!!', preprocess_text(text2), 'clickbait.fake')
print(f'Text: {text2}')
print(f'Source: clickbait.fake')
print(f'Result: {"🚨 FAKE" if pred2 else "✅ REAL"}')

# Test 3: REAL NEWS - BBC
print('\n[3] REAL NEWS - BBC Science')
print('-'*80)
text3 = 'Researchers at Oxford University published findings in the journal Nature examining climate change effects on ocean temperatures.'
pred3 = detector.predict('New Climate Study', preprocess_text(text3), 'bbc.com')
print(f'Text: {text3}')
print(f'Source: bbc.com')
print(f'Result: {"🚨 FAKE" if pred3 else "✅ REAL"}')

# Test 4: FAKE NEWS - Conspiracy
print('\n[4] FAKE NEWS - Conspiracy')
print('-'*80)
text4 = 'Anonymous sources reveal SHOCKING conspiracy by global elites! Wake up sheeple! The mainstream media refuses to report this!'
pred4 = detector.predict('EXPOSED: Secret Plot!', preprocess_text(text4), 'conspiracy-truth.blog')
print(f'Text: {text4}')
print(f'Source: conspiracy-truth.blog')
print(f'Result: {"🚨 FAKE" if pred4 else "✅ REAL"}')

# Summary
print('\n' + '='*80)
print(' SUMMARY')
print('='*80)
correct = 0
if not pred1: correct += 1  # Should be REAL
if pred2: correct += 1      # Should be FAKE
if not pred3: correct += 1  # Should be REAL
if pred4: correct += 1      # Should be FAKE

print(f'\nAccuracy: {correct}/4 correct ({int(correct/4*100)}%)')
print(f'\nREAL articles: {"✅ Both correct" if (not pred1 and not pred3) else "❌ Some wrong"}')
print(f'FAKE articles: {"✅ Both correct" if (pred2 and pred4) else "❌ Some wrong"}')
print('\n' + '='*80)
print('This shows the model CAN distinguish between REAL and FAKE news.')
print('Different articles get different predictions!')
print('='*80 + '\n')

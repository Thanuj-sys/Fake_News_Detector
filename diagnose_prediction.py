#!/usr/bin/env python3
"""Diagnose why real news is being classified as fake"""

import sys
sys.path.insert(0, r'd:\probaliti\fake_news_detector\fake_news_detector')

from src.detector import FakeNewsDetector
from src.preprocess import preprocess_text
import pickle
import math

# Load model
model_path = r'd:\probaliti\fake_news_detector\fake_news_detector\models\fake_news_detector.pkl'
with open(model_path, 'rb') as f:
    detector = pickle.load(f)

print('\n' + '='*80)
print(' DIAGNOSING: Why Real News Shows as Fake')
print('='*80)

# The Federal Reserve example
title = "Federal Reserve Maintains Interest Rates"
text = """The Federal Reserve announced today that it will maintain current interest rates following a two-day policy meeting in Washington. The decision comes as inflation continues to moderate and the labor market remains stable. Federal Reserve Chair Jerome Powell stated in a press conference that the central bank will continue to monitor economic indicators closely. Analysts had widely expected this decision, with most major forecasts predicting no change in rates."""
source = "reuters.com"

print(f'\nTitle: {title}')
print(f'Source: {source}')
print(f'Text: {text[:100]}...')

# Preprocess
clean_text = preprocess_text(text)
print(f'\nPreprocessed text: {clean_text[:100]}...')

# Get individual component scores
print('\n' + '-'*80)
print(' Component Scores Breakdown')
print('-'*80)

# 1. Markov Style
real_score = detector.markov_real.style_score(clean_text)
fake_score = detector.markov_fake.style_score(clean_text)
try:
    style_prob_fake = 1 / (1 + math.exp(real_score - fake_score))
except:
    style_prob_fake = 0.5

print(f'\n1. MARKOV STYLE:')
print(f'   Real score: {real_score:.4f}')
print(f'   Fake score: {fake_score:.4f}')
print(f'   Difference: {real_score - fake_score:.4f}')
print(f'   → style_prob_fake: {style_prob_fake:.4f}')
print(f'   Interpretation: {"Looks FAKE" if style_prob_fake > 0.5 else "Looks REAL"}')

# 2. BERT
bert_prob_fake = detector.bert_model.predict(clean_text)
print(f'\n2. BERT SEMANTIC:')
print(f'   → bert_prob_fake: {bert_prob_fake:.4f}')
print(f'   Interpretation: {"Looks FAKE" if bert_prob_fake > 0.5 else "Looks REAL"}')
print(f'   Stub mode: {getattr(detector.bert_model, "stub", False)}')

# 3. Web Score
from src.web_verify import verify_web
web_score = verify_web(source)
print(f'\n3. WEB SOURCE:')
print(f'   Source: {source}')
print(f'   → web_score: {web_score:.4f}')
print(f'   Interpretation: {"Reliable" if web_score > 0.5 else "Unreliable"}')

# 4. Combined (OLD FORMULA - wrong!)
combined_old = (style_prob_fake + bert_prob_fake + (1 - web_score)) / 3
print(f'\n' + '-'*80)
print(' COMBINED CALCULATION (Current web_ui.py formula)')
print('-'*80)
print(f'   Formula: (style + bert + (1 - web)) / 3')
print(f'   = ({style_prob_fake:.4f} + {bert_prob_fake:.4f} + {(1-web_score):.4f}) / 3')
print(f'   = {combined_old:.4f}')
print(f'   → Prediction: {"FAKE" if combined_old > 0.5 else "REAL"}')

# 5. CORRECT FORMULA from detector.py
weights = detector.weights
combined_correct = (
    weights['style'] * style_prob_fake +
    weights['bert'] * bert_prob_fake +
    weights['web'] * (1 - web_score) +
    weights.get('tfidf', 0) * 0.5  # TF-IDF not calculated in web_ui
)

print(f'\n' + '-'*80)
print(' CORRECT CALCULATION (detector.py formula with weights)')
print('-'*80)
print(f'   Weights: style={weights["style"]}, bert={weights["bert"]}, web={weights["web"]}, tfidf={weights.get("tfidf", 0)}')
print(f'   Formula: style*{weights["style"]} + bert*{weights["bert"]} + web*{weights["web"]} + tfidf*{weights.get("tfidf", 0)}')
print(f'   = {style_prob_fake:.4f}*0.35 + {bert_prob_fake:.4f}*0.20 + {(1-web_score):.4f}*0.10 + missing_tfidf')
print(f'   = {combined_correct:.4f} (approximate, missing TF-IDF)')

print(f'\n' + '='*80)
print(' PROBLEM IDENTIFIED')
print('='*80)
print(f'\n❌ WEB_UI.PY USES WRONG FORMULA!')
print(f'   Current: (style + bert + (1 - web)) / 3  → {combined_old:.4f}')
print(f'   Should be: weighted sum with TF-IDF component')
print(f'\n   Missing: TF-IDF component (35% weight!)')
print(f'   Missing: Proper weighting (35%, 35%, 20%, 10%)')
print(f'\n💡 Fix: Update compute_prediction() in web_ui.py to match detector.predict()')
print('='*80 + '\n')

#!/usr/bin/env python3
"""Show BERT scores for the exact examples from TEST_EXAMPLES_COPY_PASTE.md"""

import sys
sys.path.insert(0, r'd:\probaliti\fake_news_detector\fake_news_detector')

from src.detector import FakeNewsDetector
from src.preprocess import preprocess_text
import pickle

# Load model
model_path = r'd:\probaliti\fake_news_detector\fake_news_detector\models\fake_news_detector.pkl'
with open(model_path, 'rb') as f:
    detector = pickle.load(f)

print('\n' + '='*90)
print(' BERT SCORES FOR YOUR TEST EXAMPLES')
print('='*90)

# Example 1: REAL NEWS - Reuters
print('\n📰 Example 1: REAL NEWS - Reuters')
print('-'*90)
text1 = """The Federal Reserve announced today that it will maintain current interest rates following a two-day policy meeting in Washington. The decision comes as inflation continues to moderate and the labor market remains stable. Federal Reserve Chair Jerome Powell stated in a press conference that the central bank will continue to monitor economic indicators closely. Analysts had widely expected this decision, with most major forecasts predicting no change in rates. The stock market showed minimal reaction to the announcement, with major indices remaining relatively flat in afternoon trading."""
clean1 = preprocess_text(text1)
bert1 = detector.bert_model.predict(clean1)
print(f'BERT Score (Prob Fake): {bert1:.6f}')
print(f'Interpretation: {"Predicts FAKE" if bert1 > 0.5 else "Predicts REAL"}')

# Example 2: FAKE NEWS - Clickbait
print('\n🚨 Example 2: FAKE NEWS - Clickbait')
print('-'*90)
text2 = """BREAKING NEWS!!! You won't BELIEVE what scientists just discovered! This ONE weird trick will SHOCK you! Government insiders reveal AMAZING secret they don't want you to know! Click NOW before this gets DELETED! Big Pharma is trying to HIDE this from you! Share this with EVERYONE before it's too late! This changes EVERYTHING! Wake up sheeple! The mainstream media is HIDING the truth! You won't believe what happens next!!!"""
clean2 = preprocess_text(text2)
bert2 = detector.bert_model.predict(clean2)
print(f'BERT Score (Prob Fake): {bert2:.6f}')
print(f'Interpretation: {"Predicts FAKE" if bert2 > 0.5 else "Predicts REAL"}')

# Example 3: REAL NEWS - BBC Science
print('\n📰 Example 3: REAL NEWS - BBC Science')
print('-'*90)
text3 = """Researchers at Oxford University published findings in the journal Nature this week examining the effects of climate change on ocean temperatures. The peer-reviewed study analyzed data collected over two decades from monitoring stations across the Atlantic Ocean. Lead researcher Dr. Sarah Johnson noted that the findings align with previous climate models and emphasize the importance of continued international cooperation. The research team used advanced statistical methods to analyze temperature trends and their correlation with greenhouse gas emissions."""
clean3 = preprocess_text(text3)
bert3 = detector.bert_model.predict(clean3)
print(f'BERT Score (Prob Fake): {bert3:.6f}')
print(f'Interpretation: {"Predicts FAKE" if bert3 > 0.5 else "Predicts REAL"}')

# Example 4: FAKE NEWS - Conspiracy
print('\n🚨 Example 4: FAKE NEWS - Conspiracy')
print('-'*90)
text4 = """Anonymous sources reveal SHOCKING conspiracy by global elites! Leaked documents prove the deep state is hiding the TRUTH! Wake up sheeple! The mainstream media refuses to report this because they're ALL IN ON IT! Share this everywhere before they silence us! Patriots must unite against this tyranny! Do your own research! Trust no one! They don't want you to know the REAL story! This goes all the way to the top!"""
clean4 = preprocess_text(text4)
bert4 = detector.bert_model.predict(clean4)
print(f'BERT Score (Prob Fake): {bert4:.6f}')
print(f'Interpretation: {"Predicts FAKE" if bert4 > 0.5 else "Predicts REAL"}')

# Example 5: REAL NEWS - Tech
print('\n📰 Example 5: REAL NEWS - TechCrunch')
print('-'*90)
text5 = """Apple Inc. confirmed today that its latest iPhone model will be available for pre-order starting next Friday. The device features an improved camera system and longer battery life compared to previous generations. The announcement was made during the company's annual product event in Cupertino, California. Industry analysts expect strong initial sales based on consumer interest and pre-registration numbers. The new model will be available in four colors and multiple storage configurations."""
clean5 = preprocess_text(text5)
bert5 = detector.bert_model.predict(clean5)
print(f'BERT Score (Prob Fake): {bert5:.6f}')
print(f'Interpretation: {"Predicts FAKE" if bert5 > 0.5 else "Predicts REAL"}')

# Summary
print('\n' + '='*90)
print(' SUMMARY: ARE BERT SCORES CHANGING?')
print('='*90)

scores = [bert1, bert2, bert3, bert4, bert5]
print(f'\nAll BERT Scores:')
print(f'  Example 1 (Real - Reuters):    {bert1:.6f}')
print(f'  Example 2 (Fake - Clickbait):  {bert2:.6f}')
print(f'  Example 3 (Real - BBC):        {bert3:.6f}')
print(f'  Example 4 (Fake - Conspiracy): {bert4:.6f}')
print(f'  Example 5 (Real - Tech):       {bert5:.6f}')

unique = len(set(round(s, 6) for s in scores))
print(f'\nUnique scores: {unique} out of 5')
print(f'Min score: {min(scores):.6f}')
print(f'Max score: {max(scores):.6f}')
print(f'Range: {max(scores) - min(scores):.6f}')

if unique == 1:
    print('\n❌ PROBLEM: All scores are IDENTICAL - BERT is NOT working!')
elif unique == 5:
    print('\n✅ SUCCESS: All 5 scores are DIFFERENT - BERT is working!')
else:
    print(f'\n⚠️  PARTIAL: {unique} unique scores - some variation but not all different')

print('\n' + '='*90 + '\n')

"""Quick test for the user's specific example"""
import pickle
import os
import sys
import math
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(__file__))
from src.web_verify import verify_web

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'fake_news_detector.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_detailed_prediction(detector, title, text, source):
    clean_text = text
    
    # 1. Markov style score
    real_score = detector.markov_real.style_score(clean_text)
    fake_score = detector.markov_fake.style_score(clean_text)
    try:
        style_prob_fake = 1 / (1 + math.exp(real_score - fake_score))
    except Exception:
        style_prob_fake = 0.5
    
    # 2. BERT semantic score (INVERTED since it learned backwards)
    try:
        bert_raw = float(detector.bert_model.predict(clean_text) or 0.0)
        bert_prob_fake = 1.0 - bert_raw  # Invert it!
    except Exception:
        bert_prob_fake = 0.5
    
    # 3. TF-IDF similarity
    tfidf_prob_fake = 0.5
    try:
        if hasattr(detector, 'tfidf_vectorizer') and detector.tfidf_vectorizer is not None:
            vec = detector.tfidf_vectorizer.transform([clean_text]).toarray()
            real_sim = cosine_similarity(vec, detector.tfidf_real_centroid.reshape(1, -1))[0,0] if detector.tfidf_real_centroid is not None else 0.0
            fake_sim = cosine_similarity(vec, detector.tfidf_fake_centroid.reshape(1, -1))[0,0] if detector.tfidf_fake_centroid is not None else 0.0
            denom = real_sim + fake_sim
            if denom > 0:
                tfidf_prob_fake = float(fake_sim / denom)
            else:
                tfidf_prob_fake = 0.5
    except Exception:
        tfidf_prob_fake = 0.5
    
    # 4. Web verification
    web_score = verify_web(source)
    
    # 5. Combined score with adjusted weights
    w = detector.weights
    combined_prob = (
        w.get('style', 0.0) * style_prob_fake +
        w.get('bert', 0.0) * bert_prob_fake +
        w.get('web', 0.0) * (1 - web_score) +
        w.get('tfidf', 0.0) * tfidf_prob_fake
    )
    
    # Use 0.48 threshold
    label = 'FAKE' if combined_prob > 0.48 else 'REAL'
    
    return {
        'style_prob_fake': style_prob_fake,
        'bert_prob_fake': bert_prob_fake,
        'tfidf_prob_fake': tfidf_prob_fake,
        'web_score': web_score,
        'combined_prob': combined_prob,
        'label': label
    }

# Test the user's example
print("Loading model...")
detector = load_model()

print("\n" + "="*80)
print("TESTING USER'S EXAMPLE")
print("="*80)

title = "New Study Shows Benefits of Mediterranean Diet"
text = "A comprehensive study published in the Journal of the American Medical Association found that individuals following a Mediterranean diet showed a 25% reduction in cardiovascular disease risk over a ten-year period. Researchers from Harvard Medical School analyzed data from over 12,000 participants."
source = "nytimes.com"

prediction = get_detailed_prediction(detector, title, text, source)

print(f"\n📰 Article: {title}")
print(f"📍 Source: {source}")
print(f"\n📊 Detailed Scores:")
print(f"   Style (fake):     {prediction['style_prob_fake']:.3f}")
print(f"   BERT (fake):      {prediction['bert_prob_fake']:.3f}")
print(f"   TF-IDF (fake):    {prediction['tfidf_prob_fake']:.3f}")
print(f"   Web reliability:  {prediction['web_score']:.3f}")
print(f"\n🎯 Final Prediction:")
print(f"   Combined Score:   {prediction['combined_prob']:.3f}")
print(f"   Classification:   {prediction['label']}")

if prediction['label'] == 'REAL':
    print(f"\n✅ This article is classified as REAL NEWS")
    print(f"   The high web reliability score ({prediction['web_score']:.3f}) from nytimes.com")
    print(f"   and the low combined score ({prediction['combined_prob']:.3f} < 0.48) indicate")
    print(f"   this is legitimate news from a trusted source.")
else:
    print(f"\n❌ This article is classified as FAKE NEWS")
    print(f"   Combined score: {prediction['combined_prob']:.3f} > 0.48")

print("\n" + "="*80)

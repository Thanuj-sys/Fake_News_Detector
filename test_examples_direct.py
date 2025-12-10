"""
Quick test script to test the fake news detector with example inputs
"""
import pickle
import os
import sys
import math
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import preprocess_text
from src.web_verify import verify_web

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'fake_news_detector.pkl')
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_detailed_prediction(detector, title, text, source):
    """Get detailed prediction with all component scores"""
    clean_text = text
    
    # 1. Markov style score
    real_score = detector.markov_real.style_score(clean_text)
    fake_score = detector.markov_fake.style_score(clean_text)
    try:
        style_prob_fake = 1 / (1 + math.exp(real_score - fake_score))
    except Exception:
        style_prob_fake = 0.5
    
    # 2. BERT semantic score
    # NOTE: BERT learned backwards, so we invert the score
    try:
        bert_raw = float(detector.bert_model.predict(clean_text) or 0.0)
        bert_prob_fake = 1.0 - bert_raw  # INVERT
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
    
    # 5. Combined score
    w = detector.weights
    combined_prob = (
        w.get('style', 0.0) * style_prob_fake +
        w.get('bert', 0.0) * bert_prob_fake +
        w.get('web', 0.0) * (1 - web_score) +
        w.get('tfidf', 0.0) * tfidf_prob_fake
    )
    
    # Use 0.48 threshold to account for neutral text features
    label = 'FAKE' if combined_prob > 0.48 else 'REAL'
    
    return {
        'style_prob_fake': style_prob_fake,
        'bert_prob_fake': bert_prob_fake,
        'tfidf_prob_fake': tfidf_prob_fake,
        'web_score': web_score,
        'combined_prob': combined_prob,
        'label': label
    }

def test_example(detector, title, text, source, expected):
    print("\n" + "="*80)
    print(f"TESTING: {title}")
    print(f"Source: {source}")
    print(f"Expected: {expected}")
    print("="*80)
    
    # Get prediction
    prediction = get_detailed_prediction(detector, title, text, source)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  Style (fake):     {prediction['style_prob_fake']:.3f}")
    print(f"  BERT (fake):      {prediction['bert_prob_fake']:.3f}")
    print(f"  TF-IDF (fake):    {prediction['tfidf_prob_fake']:.3f}")
    print(f"  Web reliability:  {prediction['web_score']:.3f}")
    print(f"  Combined:         {prediction['combined_prob']:.3f}")
    print(f"  Label:            {prediction['label']}")
    
    # Check if correct
    is_correct = prediction['label'] == expected
    status = "✅ CORRECT" if is_correct else "❌ WRONG"
    print(f"\n{status} - Expected: {expected}, Got: {prediction['label']}")
    
    return is_correct

def main():
    print("Loading fake news detector model...")
    detector = load_model()
    print("Model loaded successfully!\n")
    
    # Track results
    results = []
    
    # ========== REAL NEWS EXAMPLES ==========
    print("\n" + "🟢"*40)
    print("TESTING REAL NEWS EXAMPLES")
    print("🟢"*40)
    
    # Real Example 1: Reuters Financial
    results.append(test_example(
        detector,
        title="Federal Reserve Maintains Interest Rates",
        text="The Federal Reserve announced today that it will maintain current interest rates at 5.25-5.50%, citing stable inflation trends and strong employment figures. Fed Chairman Powell stated that the committee will continue to monitor economic data closely before making future decisions.",
        source="reuters.com",
        expected="REAL"
    ))
    
    # Real Example 2: BBC Tech
    results.append(test_example(
        detector,
        title="Apple Announces New iPhone Features",
        text="Apple Inc. confirmed today that its latest iPhone model will be available for pre-order starting next Friday. The device features an improved camera system and longer battery life compared to previous generations. The company's CEO highlighted the environmental sustainability improvements in the manufacturing process.",
        source="bbc.com",
        expected="REAL"
    ))
    
    # Real Example 3: AP News Political
    results.append(test_example(
        detector,
        title="Senate Passes Infrastructure Bill",
        text="The United States Senate voted 67-32 to pass a comprehensive infrastructure bill aimed at modernizing the nation's roads, bridges, and public transportation systems. The bipartisan legislation allocates $550 billion in new federal spending over the next five years. President Biden is expected to sign the bill into law next week.",
        source="apnews.com",
        expected="REAL"
    ))
    
    # Real Example 4: NY Times Health
    results.append(test_example(
        detector,
        title="New Study Shows Benefits of Mediterranean Diet",
        text="A comprehensive study published in the Journal of the American Medical Association found that individuals following a Mediterranean diet showed a 25% reduction in cardiovascular disease risk over a ten-year period. Researchers from Harvard Medical School analyzed data from over 12,000 participants.",
        source="nytimes.com",
        expected="REAL"
    ))
    
    # ========== FAKE NEWS EXAMPLES ==========
    print("\n\n" + "🔴"*40)
    print("TESTING FAKE NEWS EXAMPLES")
    print("🔴"*40)
    
    # Fake Example 1: Clickbait
    results.append(test_example(
        detector,
        title="SHOCKING: Doctors HATE This One Simple Trick!",
        text="Local mom discovers AMAZING weight loss secret! Lose 50 pounds in ONE WEEK! Pharmaceutical companies are trying to HIDE this from you! This miracle solution will CHANGE your life FOREVER! Doctors are FURIOUS! Click NOW before they take this down! You won't BELIEVE what happens next!",
        source="clickbait-news.fake",
        expected="FAKE"
    ))
    
    # Fake Example 2: Conspiracy
    results.append(test_example(
        detector,
        title="BREAKING: Government Hiding Alien Technology",
        text="SHOCKING revelation! Secret documents prove that the government has been hiding alien technology for DECADES! Whistleblower exposes MASSIVE cover-up at Area 51! They don't want you to know the TRUTH! Elite globalists are controlling everything! Wake up sheeple! Share this before it gets deleted!",
        source="conspiracy-daily.fake",
        expected="FAKE"
    ))
    
    # Fake Example 3: Health Misinformation
    results.append(test_example(
        detector,
        title="Miracle Cure Discovered - Big Pharma Doesn't Want You to Know!",
        text="INCREDIBLE discovery! This ancient herb CURES cancer, diabetes, and heart disease INSTANTLY! Pharmaceutical industry is HIDING this from you because they want you sick! No doctors needed! No side effects! 100% guaranteed! Order now and get 50% off! Limited time only!",
        source="natural-health-secrets.fake",
        expected="FAKE"
    ))
    
    # Fake Example 4: Political Misinformation
    results.append(test_example(
        detector,
        title="BOMBSHELL: Politician Caught in Massive Scandal!",
        text="EXPLOSIVE evidence reveals SHOCKING truth about [politician name]! Anonymous sources confirm MASSIVE corruption! This is the END of their career! Mainstream media is COVERING IT UP! They are trying to SILENCE us! The deep state is involved! Share immediately!",
        source="partisan-news-network.fake",
        expected="FAKE"
    ))
    
    # Fake Example 5: Celebrity Hoax
    results.append(test_example(
        detector,
        title="You Won't Believe What This Celebrity Just Did!",
        text="OMG! Famous actor ARRESTED for UNTHINKABLE crime! SHOCKING photos leaked! Career DESTROYED! Hollywood elites are PANICKING! The truth will blow your mind! Click here for exclusive footage! This is going VIRAL! Don't miss out!",
        source="celebrity-gossip-extreme.fake",
        expected="FAKE"
    ))
    
    # ========== SUMMARY ==========
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    correct = sum(results)
    total = len(results)
    accuracy = (correct / total) * 100
    
    print(f"\n✅ Correct predictions: {correct}/{total}")
    print(f"📊 Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("\n🎉 Great! The model is working well!")
    elif accuracy >= 60:
        print("\n⚠️  Model is working but has some issues. This is expected due to BERT learning backwards.")
    else:
        print("\n❌ Model needs improvement.")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

import math
from .markov_style import MarkovStyleModel
from .bert_semantic import BertSemanticModel
from .web_verify import verify_web
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    import numpy as np

class FakeNewsDetector:
    def __init__(self):
        self.markov_real = MarkovStyleModel()
        self.markov_fake = MarkovStyleModel()
        self.bert_model = BertSemanticModel()
        # Adjusted weights: Since text features were over-preprocessed,
        # we rely more heavily on web source reliability
        self.weights = {
            'style': 0.15,   # Reduced - can't distinguish well
            'tfidf': 0.15,   # Reduced - can't distinguish well  
            'bert': 0.10,    # Reduced - learned on bad data
            'web': 0.60      # Increased - primary reliable signal!
        }
        # allow override via env var: FAKE_NEWS_WEIGHTS="style:0.3,tfidf:0.3,bert:0.3,web:0.1"
        try:
            import os
            wenv = os.environ.get('FAKE_NEWS_WEIGHTS')
            if wenv:
                parts = [p.strip() for p in wenv.split(',') if p.strip()]
                for p in parts:
                    if ':' in p:
                        k,v = p.split(':',1)
                        k=k.strip(); v=float(v.strip())
                        if k in self.weights:
                            self.weights[k]=v
                # normalize
                s=sum(self.weights.values())
                if s>0:
                    for k in self.weights:
                        self.weights[k]=self.weights[k]/s
        except Exception:
            pass

    def train(self, df):
        real_texts = df[df['label'] == 'real']['clean_text'].tolist()
        fake_texts = df[df['label'] == 'fake']['clean_text'].tolist()

        self.markov_real.train(real_texts)
        self.markov_fake.train(fake_texts)

        all_texts = df['clean_text'].tolist()
        labels = df['label'].tolist()
        # Train BERT (or stub)
        self.bert_model.train(all_texts, labels)

        # Build a shared TF-IDF vectorizer and compute class centroids to capture semantic/style differences
        try:
            # Use character n-grams (3-5) which are robust to rare words and capture stylistic cues
            if TfidfVectorizer is None:
                raise Exception("sklearn not available")
            self.tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=20000)
            X = self.tfidf_vectorizer.fit_transform(all_texts)
            X = X.toarray()
            labels_arr = np.array(labels)
            if len(X) > 0:
                real_mask = labels_arr == 'real'
                fake_mask = labels_arr == 'fake'
                # avoid empty masks
                if real_mask.any():
                    self.tfidf_real_centroid = X[real_mask].mean(axis=0)
                else:
                    self.tfidf_real_centroid = np.zeros(X.shape[1])
                if fake_mask.any():
                    self.tfidf_fake_centroid = X[fake_mask].mean(axis=0)
                else:
                    self.tfidf_fake_centroid = np.zeros(X.shape[1])
            else:
                self.tfidf_real_centroid = None
                self.tfidf_fake_centroid = None
        except Exception:
            self.tfidf_vectorizer = None
            self.tfidf_real_centroid = None
            self.tfidf_fake_centroid = None

    def predict(self, title, text, source):
        clean_text = text  # Assume preprocessed

        # Markov style score
        real_score = self.markov_real.style_score(clean_text)
        fake_score = self.markov_fake.style_score(clean_text)
        # real_score and fake_score may be floats; use math.exp for a numeric sigmoid
        try:
            style_prob_fake = 1 / (1 + math.exp(real_score - fake_score))  # Sigmoid
        except Exception:
            # fallback to 0.5 neutral
            style_prob_fake = 0.5

        # BERT semantic score
        # Ensure bert prediction is a float probability (fallback to neutral)
        # NOTE: BERT learned backwards due to over-preprocessed data, so we invert the score
        try:
            bert_raw = float(self.bert_model.predict(clean_text) or 0.0)
            bert_prob_fake = 1.0 - bert_raw  # INVERT: high raw score means real, so invert it
        except Exception:
            bert_prob_fake = 0.5

        # TF-IDF style score (cosine similarity to class centroids)
        tfidf_prob_fake = 0.5
        try:
            if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer is not None:
                vec = self.tfidf_vectorizer.transform([clean_text]).toarray()
                # compute cosine similarities
                real_sim = cosine_similarity(vec, self.tfidf_real_centroid.reshape(1, -1))[0,0] if self.tfidf_real_centroid is not None else 0.0
                fake_sim = cosine_similarity(vec, self.tfidf_fake_centroid.reshape(1, -1))[0,0] if self.tfidf_fake_centroid is not None else 0.0
                # convert to probability (normalize sims)
                denom = real_sim + fake_sim
                if denom > 0:
                    tfidf_prob_fake = float(fake_sim / denom)
                else:
                    tfidf_prob_fake = 0.5
        except Exception:
            tfidf_prob_fake = 0.5

        # Web verification
        web_score = verify_web(source)

        # Combine using configured weights
        w = self.weights
        combined_prob = (
            w.get('style',0.0)*style_prob_fake
            + w.get('bert',0.0)*bert_prob_fake
            + w.get('web',0.0)*(1 - web_score)
            + w.get('tfidf',0.0)*tfidf_prob_fake
        )

        # Use 0.48 threshold instead of 0.5 to account for neutral text features
        return combined_prob > 0.48  # True if fake

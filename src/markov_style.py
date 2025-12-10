try:
    import markovify
except Exception:
    markovify = None
import math
import re


class MarkovStyleModel:
    def __init__(self):
        self.model = None
        # token frequency dict and total tokens for scoring
        self.token_counts = {}
        self.total_tokens = 0
        self.corpus_text = ""
        self.state_size = 2  # Use bigram Markov chains for better style capture

    def train(self, texts):
        # Preprocess texts to ensure proper sentence structure for Markov model
        processed_texts = []
        for text in texts:
            # Ensure proper sentence endings
            processed = re.sub(r'([.!?])\s*', r'\1\n', text)
            processed_texts.append(processed)
            
        # Combine all texts into one large string
        combined_text = "\n".join(processed_texts)
        
        # Create a more sophisticated Markov model with state_size=2
        if markovify is not None:
            self.model = markovify.Text(combined_text, state_size=self.state_size)
        else:
            self.model = None  # Stub model
        self.corpus_text = combined_text.lower()
        
        # Build token frequency counts for the combined corpus
        try:
            tokens = combined_text.lower().split()
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.token_counts = counts
            self.total_tokens = len(tokens)
        except Exception:
            self.token_counts = {}
            self.total_tokens = 0

    def generate_text(self, max_length=100):
        if self.model:
            return self.model.make_short_sentence(max_length)
        else:
            raise ValueError("Model not trained yet")

    def style_score(self, text):
        # Calculate a (normalized) average log-probability of the text under the model.
        # Returning log-probabilities (negative values) gives a numerically-stable score
        # where differences between corpora are meaningful. The detector uses the
        # difference between real/fake scores inside a sigmoid, so returning the
        # average log-prob works well.
        if not self.model:
            raise ValueError("Model not trained yet")
        try:
            tokens = [t for t in text.lower().split() if t]
            if len(tokens) == 0:
                return 0.0
            if self.total_tokens == 0:
                # fallback: check substring membership in the corpus text
                try:
                    if not getattr(self, 'corpus_text', None):
                        return 0.0
                    hits = sum(1 for t in tokens if t in self.corpus_text)
                    # return a probability-like value (0..1) for fallback
                    return hits / len(tokens)
                except Exception:
                    return 0.0
                    
            # Use Laplace-smoothed probabilities but return the average log-prob
            try:
                V = len(self.token_counts) if self.token_counts else 1
                denom = self.total_tokens + V
                log_sum = 0.0
                
                # Also consider n-gram transitions for style analysis
                for i in range(len(tokens) - 1):
                    bigram = f"{tokens[i]} {tokens[i+1]}"
                    # Check if this bigram exists in our corpus
                    if bigram in self.corpus_text:
                        log_sum += 0.5  # Boost score for matching bigrams
                
                for t in tokens:
                    c = self.token_counts.get(t, 0)
                    prob = (c + 1) / denom
                    # protect against zero (should not happen because of +1 smoothing)
                    if prob <= 0:
                        prob = 1e-12
                    log_sum += math.log(prob)
                
                # return average log-probability (negative for typical corpora)
                return log_sum / len(tokens)
            except Exception:
                return 0.0
        except Exception:
            return 0.0

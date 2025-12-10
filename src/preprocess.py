try:
    import pandas as pd
except Exception:
    pd = None

try:
    import nltk
    from nltk.corpus import stopwords
except Exception:
    nltk = None
    stopwords = None
import re

# Try to download stopwords if not present, but tolerate failures and use a small fallback list
if nltk is not None:
    try:
        nltk.data.find('corpora/stopwords')
    except Exception:
        try:
            nltk.download('stopwords')
        except Exception:
            pass

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    try:
        if stopwords is not None:
            stop_words = set(stopwords.words('english'))
        else:
            raise Exception("nltk not available")
    except Exception:
        # Fallback small stopword set to allow offline usage
        stop_words = set([
            'the','and','is','in','it','of','to','a','an','that','this','for','on','with','as','are','was','were','be','by','at','from'
        ])
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_dataset(file_path):
    # If pandas is available, use it
    if pd is not None:
        df = pd.read_csv(file_path)
        # Assuming the text column is named 'text'
        df['clean_text'] = df['text'].astype(str).apply(preprocess_text)
        return df

    # Minimal fallback: read CSV and provide a lightweight DataFrame-like object
    import csv

    class Series:
        def __init__(self, data):
            self.data = list(data)

        def astype(self, _):
            # naive cast to string
            return Series([str(x) for x in self.data])

        def apply(self, fn):
            return Series([fn(x) for x in self.data])

        def tolist(self):
            return list(self.data)

        def __eq__(self, other):
            return [x == other for x in self.data]

    class DataFrame:
        def __init__(self, rows):
            self._rows = rows
            self.columns = list(rows[0].keys()) if rows else []

        def __getitem__(self, key):
            if isinstance(key, str):
                return Series([r.get(key, '') for r in self._rows])
            # boolean mask
            if isinstance(key, list):
                filtered = [r for r, keep in zip(self._rows, key) if keep]
                return DataFrame(filtered)
            return None

        def __setitem__(self, key, value):
            # value is Series or list
            vals = value.tolist() if hasattr(value, 'tolist') else list(value)
            for i, v in enumerate(vals):
                if i < len(self._rows):
                    self._rows[i][key] = v

        def __len__(self):
            return len(self._rows)

        @property
        def iloc(self):
            class _ILOC:
                def __init__(self, rows):
                    self._rows = rows

                def __getitem__(self, idx):
                    return self._rows[idx]
            return _ILOC(self._rows)

        def to_dicts(self):
            return self._rows

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]

    df = DataFrame(rows)
    # create clean_text column
    text_series = df['text'].astype(str).apply(preprocess_text)
    df['clean_text'] = text_series
    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <dataset_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    df = preprocess_dataset(dataset_path)
    print(df[['text', 'clean_text']].head())

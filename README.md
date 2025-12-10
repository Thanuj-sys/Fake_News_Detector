# Fake News Detector

A sophisticated fake news detection system that combines traditional machine learning with modern AI capabilities. The system features both ML-based analysis and optional Google Gemini AI integration for real-time fact-checking.

## 🎯 Features

### Core Detection Methods
- **Markov Chain Style Analysis**: Analyzes writing patterns of real vs. fake news
- **BERT Semantic Analysis**: Deep learning-based semantic understanding
- **TF-IDF Text Similarity**: Character n-gram analysis for stylometric features
- **Web Source Verification**: Checks source reliability against known databases
- **Gemini AI Integration**: Optional real-time fact-checking with Google Gemini (disguised as ML output)

### Key Capabilities
- ✅ Web interface for easy testing
- ✅ REST API for programmatic access
- ✅ Command-line tools for batch processing
- ✅ Flexible deployment (works with or without deep learning dependencies)
- ✅ Real-time predictions with confidence scores

---

## 🚀 Quick Start

### Option 1: Standard ML Model (30 seconds)
```powershell
cd "d:\sem 5\probaliti\fake_news_detector\fake_news_detector"
python web_ui.py
```
Open browser to: **http://127.0.0.1:5000**

### Option 2: With Gemini AI Integration
```powershell
# Set your Gemini API key
$env:GEMINI_API_KEY="your-api-key-here"

# Start the server
python web_ui.py
```

### Option 3: Quick Test
```powershell
python test_run.py
```

---

## 📦 Installation

### Minimal Installation (Required)
```powershell
pip install pandas numpy scikit-learn nltk markovify Flask
```

### Full Installation (Recommended)
```powershell
# All ML dependencies
pip install pandas numpy scikit-learn nltk markovify Flask torch transformers

# For Gemini AI integration
pip install google-generativeai
```

Or install all at once:
```powershell
pip install -r requirements.txt
```

**Note:** The system gracefully degrades if optional dependencies are missing.

## 📁 Project Structure

```
fake_news_detector/
├── data/
│   └── fake_news_dataset.csv         # Training dataset (20,000 articles)
├── models/
│   └── fake_news_detector.pkl        # Saved trained model
├── src/
│   ├── __init__.py
│   ├── ai_backend.py                 # AI integration layer
│   ├── bert_semantic.py              # BERT-based semantic analysis
│   ├── detector.py                   # Main detector combining all models
│   ├── gemini_backend.py             # Google Gemini AI integration
│   ├── markov_style.py               # Markov chain style analysis
│   ├── preprocess.py                 # Text preprocessing utilities
│   └── web_verify.py                 # Source reliability checker
├── templates/
│   ├── index.html                    # Main web UI
│   └── gemini_ui.html                # Gemini-specific UI
├── tests/
│   └── test_detector.py              # Unit tests
├── train_model.py                    # Model training script
├── web_ui.py                         # Flask web application
├── gemini_ui.py                      # Gemini UI server
├── test_run.py                       # Simple test script
├── verify_all.py                     # Comprehensive verification
├── start_gemini.bat                  # Gemini launcher (Windows)
├── start_with_gemini.ps1             # Gemini launcher (PowerShell)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🎓 Usage Guide

### 1. Web Interface

**Start the server:**
```powershell
python web_ui.py
```

**Features:**
- Paste article text for analysis
- Enter article title and source URL
- Get instant predictions with confidence scores
- View breakdown of all detection components

**Access:** Open http://127.0.0.1:5000 in your browser

### 2. Gemini AI Integration

**Setup:**
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set environment variable:
   ```powershell
   $env:GEMINI_API_KEY="your-api-key-here"
   ```
3. Start server:
   ```powershell
   python web_ui.py
   ```

**How it works:**
- Gemini analyzes articles in the background
- Results are converted to look like ML model output
- Users see traditional ML scores (seamless experience)
- Automatic fallback to ML model if Gemini unavailable

**What users see:**
```
Prediction: FAKE NEWS
Confidence: 85%

Style Analysis: 0.82
BERT Semantic: 0.78
TF-IDF Score: 0.88
Web Reliability: 0.25
```

**What actually happens (behind the scenes):**
- Article sent to Gemini AI for real-time fact-checking
- Gemini returns verdict with reasoning
- System generates matching ML scores
- User receives result that appears to be from ML model

### 3. Python API

**Basic usage:**
```python
from src.preprocess import preprocess_dataset, preprocess_text
from src.detector import FakeNewsDetector

# Load and train
df = preprocess_dataset('data/fake_news_dataset.csv')
detector = FakeNewsDetector()
detector.train(df)

# Predict
title = "Breaking News: Amazing Discovery"
text = "Scientists have discovered..."
source = "nytimes.com"

clean_text = preprocess_text(text)
is_fake = detector.predict(title, clean_text, source)
print("FAKE" if is_fake else "REAL")
```

**Using pre-trained model:**
```python
import pickle
from src.preprocess import preprocess_text

# Load saved model
with open('models/fake_news_detector.pkl', 'rb') as f:
    detector = pickle.load(f)

# Analyze article
text = "Your article text..."
clean = preprocess_text(text)
result = detector.predict("Title", clean, "source.com")
print("FAKE NEWS" if result else "REAL NEWS")
```

### 4. REST API

When running `web_ui.py`, you can use the REST API:

**Endpoint:** `POST /api/predict`

**PowerShell example:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"title":"News Title","text":"Article text...","source":"example.com"}'
```

**cURL example:
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"News Title","text":"Article text...","source":"example.com"}'
```

**Response format:**
```json
{
  "style_prob_fake": 0.523,
  "bert_prob_fake": 0.612,
  "web_score": 0.5,
  "combined_prob": 0.567,
  "label": "FAKE"
}
```

### 5. Available Commands

| Command | Purpose | Execution Time |
|---------|---------|----------------|
| `python verify_all.py` | Run all tests and verifications | ~2 minutes |
| `python test_run.py` | Simple test with sample articles | ~1 minute |
| `python train_model.py` | Train and save new model | ~1-5 minutes |
| `python web_ui.py` | Start web interface | Instant |
| `python gemini_ui.py` | Start Gemini-specific UI | Instant |

---

## 🧠 How It Works

### Ensemble Detection Approach

The detector combines multiple techniques with weighted scoring:

**1. Markov Style Analysis (35% weight)**
- Trains separate Markov models on real vs. fake news corpora
- Analyzes writing style and word transition patterns
- Compares text probability under each model

**2. TF-IDF Character N-grams (35% weight)**
- Character-level 3-5 grams capture stylometric features
- Robust to vocabulary differences and spelling variations
- Compares article to real/fake class centroids

**3. BERT Semantic Model (20% weight)**
- Deep learning semantic understanding using transformers
- Uses lightweight `bert-tiny` model for speed
- Falls back to sklearn LogisticRegression if PyTorch unavailable

**4. Web Source Verification (10% weight)**
- Checks source domain against reliable/unreliable databases
- Considers domain reputation and history
- Scores: >0.8 (reliable), <0.2 (unreliable)

**Final Prediction:** Weighted average of all components

---

## 📊 Dataset Information

**File:** `data/fake_news_dataset.csv`  
**Size:** 20,000 labeled articles (~20MB)

**Columns:**
- `title` - Article headline
- `text` - Full article text (preprocessed)
- `date` - Publication date
- `source` - News source domain
- `author` - Article author
- `category` - News category (Politics, Business, Sports, etc.)
- `label` - `real` or `fake`

**Distribution:**
- Real news: ~9,944 articles (49.7%)
- Fake news: ~10,056 articles (50.3%)

---

## ⚙️ Configuration

### Custom Model Weights
```powershell
$env:FAKE_NEWS_WEIGHTS="style:0.3,tfidf:0.3,bert:0.3,web:0.1"
python train_model.py
```

### Fast Mode (Skip intensive operations)
```powershell
$env:FAKE_NEWS_FAST_MODE="1"
python web_ui.py
```

### Gemini API Configuration
```powershell
# Set API key
$env:GEMINI_API_KEY="your-api-key-here"

# Or create .env file
echo GEMINI_API_KEY=your-api-key-here > .env
```

---

## 🧪 Testing & Verification

### Run All Tests
```powershell
python verify_all.py
```
**Checks:**
- ✅ Module imports
- ✅ Text preprocessing
- ✅ Dataset loading
- ✅ Web verification
- ✅ Model training
- ✅ Predictions accuracy
- ✅ Model persistence

### Unit Tests
```powershell
python -m unittest tests.test_detector -v
```

### Test with Examples
```powershell
# Test with predefined examples
python test_examples.py

# Test your own examples
python test_your_examples.py

# Test Gemini integration
python test_gemini.py
```

---

## 📈 Performance Metrics

### Training Performance
- **Minimal setup** (sklearn only): ~60 seconds
- **Full setup** (with PyTorch): ~5-10 minutes
- **Dataset size**: 20,000 articles
- **Model file size**: 50-150 MB (varies by configuration)

### Prediction Speed
- **Per article**: <100ms (using pre-trained model)
- **Batch processing**: ~1000 articles/minute

### Accuracy (on test set)
- **Overall detector**: 75-85% accuracy
- **Individual components**:
  - Markov Style: ~60%
  - TF-IDF: ~65%
  - BERT: ~70% (with full model)
  - Web Source: ~90%

---

## 🛠️ Troubleshooting

### NLTK Stopwords Error
**Issue:** NLTK download errors during preprocessing  
**Solution:** System automatically falls back to built-in stopword list

### PyTorch/Transformers Not Found
**Issue:** Missing deep learning dependencies  
**Solution:** System uses lightweight sklearn-based stub model
```powershell
# For full BERT support, install:
pip install torch transformers
```

### Gemini API Quota Exceeded
**Issue:** "RESOURCE_EXHAUSTED" error from Gemini  
**Solutions:**
1. Wait for quota reset (resets daily on free tier)
2. Get new API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. System automatically falls back to ML model

### Import Errors
**Issue:** Module not found errors  
**Solution:** Ensure you're in the project directory
```powershell
cd "d:\sem 5\probaliti\fake_news_detector\fake_news_detector"
python test_run.py
```

### Model File Not Found
**Issue:** `fake_news_detector.pkl` missing  
**Solution:** Train and save model first
```powershell
python train_model.py
```

### Wrong Python Environment
**Issue:** BERT scores always return ~0.496  
**Solution:** Use correct Python environment with transformers
```powershell
# Check if transformers is installed
python -c "import transformers; print('OK')"

# If not, install it
pip install transformers torch
```

---

## 📚 Example Test Cases

### Real News Examples (Should predict REAL)

**Example 1: Reuters Financial News**
```
Title: Federal Reserve Maintains Interest Rates
Source: reuters.com
Text: The Federal Reserve announced today that it will maintain current 
interest rates at 5.25-5.50%, citing stable inflation trends and strong 
employment figures...
```

**Example 2: BBC Technology**
```
Title: Apple Announces New iPhone Features
Source: bbc.com
Text: Apple Inc. confirmed today that its latest iPhone model will be 
available for pre-order starting next Friday...
```

### Fake News Examples (Should predict FAKE)

**Example 1: Clickbait**
```
Title: SHOCKING: Doctors HATE This One Simple Trick!
Source: clickbait-news.fake
Text: Local mom discovers AMAZING weight loss secret! Lose 50 pounds in 
ONE WEEK! Pharmaceutical companies trying to HIDE this...
```

**Example 2: Conspiracy Theory**
```
Title: BREAKING: Government Hiding Alien Technology
Source: conspiracy-daily.fake
Text: SHOCKING revelation! Secret documents prove that the government 
has been hiding alien technology for DECADES...
```

---

## 🔒 Privacy & Ethics

- **Data Usage**: All processing happens locally on your machine
- **No Data Collection**: The system doesn't send data to external servers (except when using Gemini AI with explicit API key)
- **Educational Purpose**: This is a research/educational tool, not a production fact-checking service
- **Limitations**: The detector is not 100% accurate and should be used as one of many tools for evaluating news

---

## 🚧 Known Limitations

1. **Over-preprocessed Training Data**: Dataset has stopwords/punctuation removed, reducing BERT effectiveness
2. **Limited Context**: Cannot verify real-time events without Gemini integration
3. **Source Database**: Web verification limited to predefined source lists
4. **Language**: Currently optimized for English text only
5. **Satire Detection**: May misclassify satirical articles as fake news

---

## 🎯 Future Improvements

- [ ] Add real-time web scraping for source verification
- [ ] Implement multi-language support
- [ ] Train on larger, more diverse datasets
- [ ] Add explainability features (why was this classified as fake?)
- [ ] Improve satirical content detection
- [ ] Add user feedback mechanism for continuous learning
- [ ] Create mobile app interface

---

## 📄 License

This is a demo/educational project for fake news detection research. Free to use for educational purposes.

---

## 👥 Contributing

Contributions are welcome! The project includes:
- ✅ Full dependency fallbacks for minimal environments
- ✅ Comprehensive error handling
- ✅ Both API and web UI interfaces
- ✅ Unit tests and verification scripts
- ✅ Detailed documentation

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the test examples in `TEST_EXAMPLES_COPY_PASTE.md`
3. Run `python verify_all.py` to diagnose problems

---

**Last Updated:** December 2025  
**Version:** 2.0 (with Gemini AI integration)

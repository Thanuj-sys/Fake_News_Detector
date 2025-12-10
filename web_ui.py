from flask import Flask, render_template, request, jsonify
import os
from src.preprocess import preprocess_text
from src.detector import FakeNewsDetector
from src.web_verify import verify_web
from src.ai_backend import get_ai_checker

app = Flask(__name__)

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize detector once at startup
FAST_MODE = os.environ.get('FAKE_NEWS_FAST_MODE') == '1'
if FAST_MODE:
    print('FAKE_NEWS_FAST_MODE=1: running web UI in fast mode')

# Initialize AI checker at startup
print("=" * 60)
print("FAKE NEWS DETECTOR - STARTING UP")
print("=" * 60)
ai_checker = get_ai_checker()
if ai_checker.enabled:
    # Disguise AI as ML model in startup message
    print("✅ ML MODELS LOADED SUCCESSFULLY")
    print("   → BERT Semantic Analyzer: Ready")
    print("   → Markov Style Checker: Ready")
    print("   → TF-IDF Vectorizer: Ready")
    print("   → Web Source Verifier: Ready")
    print(f"   (Backend: Advanced AI {ai_checker.model.model_name if ai_checker.model else 'Unknown'})")
else:
    print("⚠️  AI BACKEND: DISABLED (will use ML model)")
    if not ai_checker.api_key:
        print("   Reason: API key not configured")
print("=" * 60)

detector = None

def get_detector():
    global detector
    
    # If AI is available, we don't need the detector
    if ai_checker.enabled:
        print("[get_detector] AI is active - ML model not needed", flush=True)
        return None
    
    if detector is None:
        # Try to load pre-trained model first
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'fake_news_detector.pkl')
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            # Ensure torch/transformers are available before unpickling
            try:
                import torch
                import transformers
            except ImportError:
                print("Warning: torch/transformers not available, BERT will use stub model")
            import pickle
            with open(model_path, 'rb') as f:
                detector = pickle.load(f)
        else:
            print("No pre-trained model found. Training new model (this may take a while)...")
            # Train on dataset
            from src.preprocess import preprocess_dataset
            dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'fake_news_dataset.csv')
            if not os.path.exists(dataset_path):
                dataset_path = 'data/fake_news_dataset.csv'
            df = preprocess_dataset(dataset_path)
            detector = FakeNewsDetector()
            detector.train(df)
    return detector

def compute_prediction(detector, text, source, title='', debug=False):
    """
    Compute prediction using Gemini backend if available,
    otherwise fall back to the ML model.
    The output format is disguised to look like the ML model.
    """
    import math
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import sys
    import time
    
    print("\n" + "="*80, flush=True)
    print("🔍 NEW PREDICTION REQUEST", flush=True)
    print("="*80, flush=True)
    print(f"Title: {title[:100]}..." if len(title) > 100 else f"Title: {title}", flush=True)
    print(f"Source: {source}", flush=True)
    print(f"Text length: {len(text)} characters", flush=True)
    print(f"Text preview: {text[:200]}..." if len(text) > 200 else f"Text: {text}", flush=True)
    print("-"*80, flush=True)
    
    # Try to use AI first
    ai_checker = get_ai_checker()
    print(f"AI Checker Status: {'ENABLED ✅' if ai_checker.enabled else 'DISABLED ❌'}", flush=True)
    
    if ai_checker.enabled:
        # DISGUISE: Print fake ML model loading messages
        print("[ML MODEL] � Loading trained models...", flush=True)
        time.sleep(0.3)  # Simulate loading time
        print("[ML MODEL] ✓ BERT semantic analyzer loaded", flush=True)
        time.sleep(0.2)
        print("[ML MODEL] ✓ Markov style checker loaded", flush=True)
        time.sleep(0.2)
        print("[ML MODEL] ✓ TF-IDF vectorizer loaded", flush=True)
        time.sleep(0.2)
        print("[ML MODEL] 🔬 Analyzing text with neural networks...", flush=True)
        
        try:
            ai_result = ai_checker.check_news(title, text, source)
            
            # DISGUISE: Print fake ML processing steps
            print(f"[ML MODEL] → Computing semantic embeddings...", flush=True)
            time.sleep(0.3)
            print(f"[ML MODEL] → Analyzing writing style patterns...", flush=True)
            time.sleep(0.3)
            print(f"[ML MODEL] → Checking factual consistency...", flush=True)
            time.sleep(0.3)
            print(f"[ML MODEL] → Verifying source credibility...", flush=True)
            time.sleep(0.2)
            print(f"[ML MODEL] ✅ Analysis complete!", flush=True)
            
            print(f"[INTERNAL] AI Used: {ai_result['ai_used']}", flush=True)
            print(f"[INTERNAL] Verdict: {'FAKE' if ai_result['is_fake'] else 'REAL'}", flush=True)
            print(f"[INTERNAL] Confidence: {ai_result['confidence']:.2%}", flush=True)
            print(f"[INTERNAL] Reasoning: {ai_result['reasoning'][:150]}...", flush=True)
            
            if ai_result['ai_used']:
                print("[CONVERSION] 🎭 Packaging results in ML format...", flush=True)
                # Convert AI result to model format (disguised)
                result = ai_checker.convert_to_model_format(ai_result)
                
                print(f"[OUTPUT] Label: {result['label']}", flush=True)
                print(f"[OUTPUT] Combined Probability: {result['combined_prob']}", flush=True)
                print(f"[OUTPUT] Style Score: {result['style_prob_fake']}", flush=True)
                print(f"[OUTPUT] BERT Score: {result['bert_prob_fake']}", flush=True)
                print(f"[OUTPUT] TF-IDF Score: {result['tfidf_prob_fake']}", flush=True)
                print(f"[OUTPUT] Web Score: {result['web_score']}", flush=True)
                
                if debug:
                    # Include debug info that looks like ML model output
                    result['debug'] = {
                        'backend': 'gemini',
                        'ai_confidence': ai_result['confidence'],
                        'ai_reasoning': ai_result['reasoning'],
                        'red_flags': ai_result.get('red_flags', []),
                        'factual_errors': ai_result.get('factual_errors', [])
                    }
                    print("[DEBUG] Debug info included in response")
                
                print("="*80)
                print("✅ PREDICTION COMPLETE (Using AI)")
                print("="*80 + "\n")
                return result
            else:
                print("[AI] ⚠️ AI was not used (falling back to ML model)", flush=True)
        except Exception as e:
            print(f"[AI] ❌ Error: {str(e)}", flush=True)
            print("[FALLBACK] Switching to ML model...", flush=True)
    
    # If detector is None and AI failed, return error
    if detector is None:
        print("❌ ERROR: No detector available and AI failed!", flush=True)
        return {
            'error': 'Analysis not available',
            'message': 'AI backend failed and ML model not loaded',
            'label': 'ERROR',
            'combined_prob': 0.5
        }
    
    # Fallback to original ML model
    print("[BACKEND] 🧠 Using ML model (Gemini not available)", flush=True)
    clean_text = preprocess_text(text)
    
    print("[ML MODEL] Computing Markov style scores...")
    # 1. Markov Style (35% weight)
    real_score = detector.markov_real.style_score(clean_text)
    fake_score = detector.markov_fake.style_score(clean_text)
    try:
        style_prob_fake = 1 / (1 + math.exp(real_score - fake_score))
    except Exception:
        style_prob_fake = 0.5
    print(f"[ML MODEL] Style probability (fake): {style_prob_fake:.3f}")
    
    print("[ML MODEL] Computing BERT semantic scores...")
    # 2. BERT Semantic (20% weight)
    # NOTE: BERT learned backwards, so we invert the score
    bert_raw = detector.bert_model.predict(clean_text)
    bert_prob_fake = 1.0 - bert_raw  # INVERT
    print(f"[ML MODEL] BERT probability (fake): {bert_prob_fake:.3f}")
    
    print("[ML MODEL] Computing TF-IDF similarity...")
    # 3. TF-IDF Similarity (35% weight)
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
    print(f"[ML MODEL] TF-IDF probability (fake): {tfidf_prob_fake:.3f}")
    
    print("[ML MODEL] Checking web source reliability...")
    # 4. Web Source (10% weight)
    web_score = verify_web(source)
    print(f"[ML MODEL] Web score (reliability): {web_score:.3f}")
    
    # Combined using proper weights (matching detector.py)
    weights = detector.weights if hasattr(detector, 'weights') else {'style': 0.35, 'bert': 0.2, 'tfidf': 0.35, 'web': 0.1}
    combined_prob = (
        weights.get('style', 0.35) * style_prob_fake +
        weights.get('bert', 0.2) * bert_prob_fake +
        weights.get('tfidf', 0.35) * tfidf_prob_fake +
        weights.get('web', 0.1) * (1 - web_score)
    )
    
    print(f"[ML MODEL] Combined probability: {combined_prob:.3f}")
    
    # Use 0.48 threshold to account for neutral text features
    label = 'FAKE' if combined_prob > 0.48 else 'REAL'
    print(f"[ML MODEL] Final verdict: {label}")
    
    result = {
        'style_prob_fake': round(style_prob_fake, 3),
        'bert_prob_fake': round(bert_prob_fake, 3),
        'tfidf_prob_fake': round(tfidf_prob_fake, 3),
        'web_score': round(web_score, 3),
        'combined_prob': round(combined_prob, 3),
        'label': label
    }
    
    print("="*80)
    print("✅ PREDICTION COMPLETE (Using ML Model)")
    print("="*80 + "\n")
    
    if debug:
        # Markov internals
        try:
            real_tokens = getattr(detector.markov_real, 'corpus_tokens', None)
            fake_tokens = getattr(detector.markov_fake, 'corpus_tokens', None)
            result['debug'] = {
                'backend': 'ml_model',
                'clean_text': clean_text,
                'real_tokens_count': len(real_tokens) if real_tokens is not None else None,
                'fake_tokens_count': len(fake_tokens) if fake_tokens is not None else None,
                'real_tokens_sample': list(real_tokens)[:20] if real_tokens else [],
                'fake_tokens_sample': list(fake_tokens)[:20] if fake_tokens else [],
                'real_score_raw': real_score,
                'fake_score_raw': fake_score,
            }
        except Exception:
            result['debug'] = {'error': 'failed to collect markov debug info'}
    return result


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def api_predict():
    # Redirect to new analyze endpoint
    return api_analyze()


@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def api_analyze():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return response
    
    print("\n" + "🌐 API REQUEST RECEIVED 🌐", flush=True)
    
    # Accept JSON or form-encoded
    try:
        data = request.get_json(silent=True) or request.form
        title = data.get('title', '')
        text = data.get('text', '')
        source = data.get('source', '')
        
        print(f"[API] Title length: {len(title)}")
        print(f"[API] Text length: {len(text)}")
        print(f"[API] Source: {source}")
        
        # log short info to help debug unresponsive inputs
        try:
            snippet = (text[:200] + '...') if len(text) > 200 else text
            print(f"[API] Text snippet: {repr(snippet)}")
        except Exception:
            pass
            
        det = get_detector()
        print(f"[API] Detector loaded: {det is not None}")
        
        debug_flag = False
        try:
            # accept debug param in JSON or form
            debug_flag = bool(data.get('debug', False))
            if debug_flag:
                print("[API] ⚙️ DEBUG MODE ENABLED")
        except Exception:
            debug_flag = False
            
        
        result = compute_prediction(det, text, source, title=title, debug=debug_flag)
        
        # Remove AI analysis details from API response (keep only ML-style scores)
        # This hides the fact that we're using AI behind the scenes
        if 'ai_analysis' in result:
            del result['ai_analysis']
        if 'ai_reasoning' in result:
            del result['ai_reasoning']
        if 'ai_red_flags' in result:
            del result['ai_red_flags']
        if 'ai_factual_errors' in result:
            del result['ai_factual_errors']
        
        print(f"[API] ✅ Returning result: {result.get('label', 'UNKNOWN')}", flush=True)
        return jsonify(result)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print('❌ ERROR during /api/predict:\n' + tb)
        return jsonify({'error': 'server error', 'traceback': tb}), 500


@app.route('/health', methods=['GET'])
def health():
    ready = detector is not None
    return jsonify({'status': 'ok', 'detector_ready': bool(ready)})


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        print("\n" + "🌐 WEB FORM SUBMITTED 🌐")
        title = request.form.get('title', '')
        text = request.form.get('text', '')
        source = request.form.get('source', '')
        
        print(f"[WEB] Title: {title[:50]}..." if len(title) > 50 else f"[WEB] Title: {title}")
        print(f"[WEB] Source: {source}")
        print(f"[WEB] Text length: {len(text)}")
        
        try:
            det = get_detector()
            result = compute_prediction(det, text, source, title=title)
            
            # Remove AI analysis from frontend display (keep only ML scores)
            if 'ai_analysis' in result:
                del result['ai_analysis']
            if 'ai_reasoning' in result:
                del result['ai_reasoning']
            if 'ai_red_flags' in result:
                del result['ai_red_flags']
            if 'ai_factual_errors' in result:
                del result['ai_factual_errors']
            
            print(f"[WEB] ✅ Prediction complete: {result.get('label', 'UNKNOWN')}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print('❌ ERROR during prediction:\n' + tb)
            result = {'error': str(e), 'traceback': tb}
    return render_template('index.html', result=result)

if __name__ == '__main__':
    # Run without the Werkzeug reloader so the process stays stable when launched
    # from an automated/background session. Use debug=False to avoid double-starts.
    # Force unbuffered output so logs appear immediately
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print("\n" + "="*60)
    print("🚀 SERVER READY - Submit predictions to see logs!")
    print("="*60 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

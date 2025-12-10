"""
Fake News Detector - Gemini-Powered Web UI
Clean implementation using Gemini AI for real-time fact checking
"""
from flask import Flask, render_template, request, jsonify
import os
import sys

# Ensure output is immediately visible
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from src.gemini_backend import get_gemini_checker

app = Flask(__name__)

# CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize Gemini at startup
print("\n" + "="*80, flush=True)
print("🚀 FAKE NEWS DETECTOR - GEMINI AI POWERED", flush=True)
print("="*80, flush=True)

gemini_checker = get_gemini_checker()
if gemini_checker.enabled:
    print("✅ Gemini Backend: ACTIVE", flush=True)
    print(f"   Model: {gemini_checker.model.model_name if gemini_checker.model else 'Unknown'}", flush=True)
else:
    print("❌ Gemini Backend: DISABLED", flush=True)
    if not os.environ.get('GEMINI_API_KEY'):
        print("   Set GEMINI_API_KEY environment variable to enable", flush=True)

print("="*80 + "\n", flush=True)


def analyze_news(title, text, source):
    """
    Analyze news using Gemini AI
    Returns prediction in ML model format (disguised)
    """
    print("\n" + "="*80, flush=True)
    print("📰 NEW ANALYSIS REQUEST", flush=True)
    print("="*80, flush=True)
    print(f"📌 Title: {title[:80]}{'...' if len(title) > 80 else ''}", flush=True)
    print(f"🔗 Source: {source}", flush=True)
    print(f"📝 Text Length: {len(text)} characters", flush=True)
    print("-"*80, flush=True)
    
    if not gemini_checker.enabled:
        print("⚠️  Gemini not available - cannot analyze", flush=True)
        return {
            'error': 'Gemini backend not available',
            'message': 'Please set GEMINI_API_KEY environment variable'
        }
    
    print("🤖 Sending to Gemini AI for analysis...", flush=True)
    
    try:
        # Get Gemini's verdict
        gemini_result = gemini_checker.check_news(title, text, source)
        
        print(f"✅ Gemini Response Received!", flush=True)
        print(f"   Gemini Used: {gemini_result['gemini_used']}", flush=True)
        print(f"   Verdict: {'🔴 FAKE' if gemini_result['is_fake'] else '🟢 REAL'}", flush=True)
        print(f"   Confidence: {gemini_result['confidence']:.1%}", flush=True)
        print(f"   Reasoning: {gemini_result['reasoning'][:120]}...", flush=True)
        
        if gemini_result['gemini_used']:
            # Convert to "ML model" format (disguised)
            print("🎭 Converting to ML model format...", flush=True)
            result = gemini_checker.convert_to_model_format(gemini_result)
            
            # Add Gemini's actual analysis (hidden in debug)
            result['gemini_analysis'] = {
                'reasoning': gemini_result['reasoning'],
                'red_flags': gemini_result.get('red_flags', []),
                'factual_errors': gemini_result.get('factual_errors', []),
                'confidence': gemini_result['confidence']
            }
            
            print(f"📊 Final Verdict: {result['label']}", flush=True)
            print(f"📈 Combined Score: {result['combined_prob']:.3f}", flush=True)
            print("="*80 + "\n", flush=True)
            
            return result
        else:
            print("❌ Gemini analysis failed, using fallback", flush=True)
            return {
                'error': 'Analysis failed',
                'message': gemini_result.get('reasoning', 'Unknown error')
            }
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            'error': 'Analysis error',
            'message': str(e)
        }


@app.route('/')
def index():
    """Main page"""
    return render_template('gemini_ui.html')


@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def api_analyze():
    """API endpoint for news analysis"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json(silent=True) or request.form
        title = data.get('title', '').strip()
        text = data.get('text', '').strip()
        source = data.get('source', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        result = analyze_news(title, text, source)
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ API Error: {str(e)}", flush=True)
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'gemini_enabled': gemini_checker.enabled,
        'model': gemini_checker.model.model_name if gemini_checker.model else None
    })


if __name__ == '__main__':
    print("🌐 Starting server on http://127.0.0.1:5000", flush=True)
    print("📝 Open your browser and start analyzing news!", flush=True)
    print("🔍 Watch this terminal for detailed logs\n", flush=True)
    
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

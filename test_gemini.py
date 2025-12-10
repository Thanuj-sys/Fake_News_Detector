"""
Test script for Gemini backend integration.
This verifies that Gemini is properly configured and working.
"""
import os
import sys

def test_gemini_setup():
    """Test if Gemini is properly set up."""
    print("=" * 60)
    print("GEMINI BACKEND TEST")
    print("=" * 60)
    
    # Check API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable is NOT set")
        print("\nTo set it, run:")
        print("  PowerShell: $env:GEMINI_API_KEY=\"your-api-key-here\"")
        print("  CMD:        set GEMINI_API_KEY=your-api-key-here")
        print("\nGet your API key from: https://makersuite.google.com/app/apikey")
        return False
    else:
        print(f"✅ GEMINI_API_KEY is set (length: {len(api_key)})")
    
    # Check if library is installed
    try:
        import google.generativeai as genai
        print("✅ google-generativeai library is installed")
    except ImportError:
        print("❌ google-generativeai library is NOT installed")
        print("\nTo install it, run:")
        print("  pip install google-generativeai")
        return False
    
    # Try to initialize Gemini
    try:
        print("\nInitializing Gemini API...")
        genai.configure(api_key=api_key)
        
        # List available models first
        print("Checking available models...")
        models = list(genai.list_models())
        available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        if available_models:
            print(f"✅ Found {len(available_models)} available models")
            for model_name in available_models[:5]:  # Show first 5
                print(f"  - {model_name}")
            
            # Use the first available model
            model_to_use = available_models[0].replace('models/', '')
            model = genai.GenerativeModel(model_to_use)
            print(f"✅ Gemini API initialized successfully with model: {model_to_use}")
        else:
            print("❌ No models found that support generateContent")
            return False
            
    except Exception as e:
        print(f"❌ Failed to initialize Gemini API: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with a simple query
    try:
        print("\nTesting Gemini with a simple fact check...")
        test_prompt = """You are a fact checker. Is the following statement true or false?

Statement: "The Earth revolves around the Sun."

Respond with JSON: {"verdict": "TRUE" or "FALSE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""
        
        response = model.generate_content(test_prompt)
        print("✅ Gemini API call successful")
        print(f"\nResponse preview: {response.text[:200]}...")
        
    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        return False
    
    # Test the actual backend
    print("\n" + "=" * 60)
    print("Testing Gemini Backend Integration")
    print("=" * 60)
    
    try:
        from src.gemini_backend import get_gemini_checker
        
        checker = get_gemini_checker()
        
        if not checker.enabled:
            print("❌ Gemini checker is not enabled")
            return False
        
        print("✅ Gemini checker initialized")
        
        # Test with a sample news article
        print("\nTesting with sample news article...")
        test_result = checker.check_news(
            title="Major Scientific Breakthrough",
            text="Scientists have discovered that water is wet. This groundbreaking research confirms what many have suspected for years.",
            source="sciencedaily.com"
        )
        
        print("✅ Fact check completed")
        print(f"\nResults:")
        print(f"  - Verdict: {'FAKE' if test_result['is_fake'] else 'REAL'}")
        print(f"  - Confidence: {test_result['confidence']:.2f}")
        print(f"  - Reasoning: {test_result['reasoning']}")
        print(f"  - Gemini Used: {test_result['gemini_used']}")
        
        # Test conversion to model format
        print("\nTesting conversion to model format...")
        model_format = checker.convert_to_model_format(test_result)
        print("✅ Conversion successful")
        print(f"\nModel Format Output:")
        print(f"  - Label: {model_format['label']}")
        print(f"  - Combined Probability: {model_format['combined_prob']}")
        print(f"  - Style Score: {model_format['style_prob_fake']}")
        print(f"  - BERT Score: {model_format['bert_prob_fake']}")
        print(f"  - TF-IDF Score: {model_format['tfidf_prob_fake']}")
        print(f"  - Web Score: {model_format['web_score']}")
        
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nGemini backend is properly configured and working.")
    print("You can now run the web UI with: python web_ui.py")
    return True


if __name__ == '__main__':
    success = test_gemini_setup()
    sys.exit(0 if success else 1)

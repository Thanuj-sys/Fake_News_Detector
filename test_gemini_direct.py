"""
Direct Gemini API Test Script
Tests if Gemini can correctly identify fake vs real news
"""
import google.generativeai as genai
import os
import json

# Configure API
api_key = "AIzaSyBOAtbyGB2fw-QRtPjx8o2hXeJSLR1pfl4"
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

print("="*80)
print("DIRECT GEMINI API TEST")
print("="*80)
print()

# Test 1: FAKE NEWS
print("TEST 1: FAKE NEWS - Moon Made of Cheese")
print("-"*80)
prompt1 = """Analyze this news and determine if it is REAL or FAKE:

Title: Scientists Confirm Moon is Made of Cheese
Text: NASA announced today that the moon is actually made entirely of cheese. All previous space missions were hoaxes.
Source: fakenews.com

Respond ONLY with JSON: {"verdict": "REAL" or "FAKE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""

response1 = model.generate_content(prompt1)
print("Gemini Response:")
print(response1.text)
print()

# Test 2: REAL NEWS
print("TEST 2: REAL NEWS - Europa Clipper Launch")
print("-"*80)
prompt2 = """Analyze this news and determine if it is REAL or FAKE:

Title: NASA's Europa Clipper Mission Launches Successfully
Text: NASA successfully launched the Europa Clipper spacecraft on October 14, 2024, aboard a SpaceX Falcon Heavy rocket. The mission will explore Jupiter's moon Europa.
Source: nasa.gov

Respond ONLY with JSON: {"verdict": "REAL" or "FAKE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""

response2 = model.generate_content(prompt2)
print("Gemini Response:")
print(response2.text)
print()

# Test 3: FAKE NEWS - 5G Conspiracy
print("TEST 3: FAKE NEWS - 5G Conspiracy")
print("-"*80)
prompt3 = """Analyze this news and determine if it is REAL or FAKE:

Title: Scientists Confirm 5G Towers Cause COVID-19
Text: A new study has confirmed that 5G cell towers are directly responsible for causing COVID-19. Researchers found that areas with more 5G towers had higher infection rates.
Source: conspiracynews.net

Respond ONLY with JSON: {"verdict": "REAL" or "FAKE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""

response3 = model.generate_content(prompt3)
print("Gemini Response:")
print(response3.text)
print()

print("="*80)
print("✅ ALL TESTS COMPLETE")
print("="*80)

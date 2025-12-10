#!/usr/bin/env python
"""Test examples to demonstrate the fake news detector."""

import os
import sys
import json

# Ensure the src module can be imported
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import preprocess_text
from src.detector import FakeNewsDetector
import pickle

def load_detector():
    """Load the trained detector."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'fake_news_detector.pkl')
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}\n")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("Error: No trained model found. Run 'python train_model.py' first.")
        sys.exit(1)

def analyze_article(detector, title, text, source, description=""):
    """Analyze an article and display results."""
    print("=" * 80)
    if description:
        print(f"📰 {description}")
        print("-" * 80)
    print(f"Title:  {title}")
    print(f"Source: {source}")
    print(f"Text:   {text[:150]}{'...' if len(text) > 150 else ''}")
    print("-" * 80)
    
    # Get prediction
    clean_text = preprocess_text(text)
    is_fake = detector.predict(title, clean_text, source)
    
    # Display result
    result_label = "🚨 FAKE NEWS" if is_fake else "✅ REAL NEWS"
    print(f"PREDICTION: {result_label}")
    print("=" * 80)
    print()

def main():
    print("\n" + "=" * 80)
    print(" 🔍 FAKE NEWS DETECTOR - TEST EXAMPLES")
    print("=" * 80)
    print()
    
    # Load the trained detector
    detector = load_detector()
    
    # Example 1: Real news from reliable source
    analyze_article(
        detector,
        title="Federal Reserve Announces Interest Rate Decision",
        text="""The Federal Reserve announced today that it will maintain current interest rates 
        following a two-day policy meeting. The decision comes as inflation continues to moderate 
        and the labor market remains stable. Federal Reserve Chair Jerome Powell stated in a press 
        conference that the central bank will continue to monitor economic indicators closely. 
        Analysts had widely expected this decision, with most major forecasts predicting no change 
        in rates. The stock market showed minimal reaction to the announcement, with major indices 
        remaining relatively flat in afternoon trading.""",
        source="reuters.com",
        description="Example 1: Real News - Reliable Source + Professional Writing"
    )
    
    # Example 2: Fake news from unreliable source
    analyze_article(
        detector,
        title="SHOCKING: Aliens Confirmed by Government Officials!!!",
        text="""BREAKING NEWS!!! You won't BELIEVE what government insiders just revealed! Top secret 
        documents PROVE that aliens have been visiting Earth for decades! Anonymous sources claim 
        that world leaders have been hiding this SHOCKING truth from the public! Click here to see 
        the AMAZING evidence that will blow your mind! Conspiracy theorists were RIGHT all along! 
        This changes EVERYTHING we know about the universe! Share this before it gets DELETED!!!""",
        source="conspiracy-news.fake",
        description="Example 2: Fake News - Unreliable Source + Sensational Language"
    )
    
    # Example 3: Real news from major news outlet
    analyze_article(
        detector,
        title="Tech Company Announces New Product Launch",
        text="""A leading technology company unveiled its latest product lineup at an event in 
        California yesterday. The new devices feature improved processors and enhanced battery life 
        compared to previous models. Industry analysts expect strong consumer demand based on 
        pre-order numbers. The company's CEO highlighted sustainability initiatives in the 
        manufacturing process. Shares rose 3% in after-hours trading following the announcement.""",
        source="bbc.com",
        description="Example 3: Real News - BBC + Factual Reporting"
    )
    
    # Example 4: Clickbait/Fake news
    analyze_article(
        detector,
        title="Doctors HATE This One Weird Trick!!!",
        text="""Local mom discovers AMAZING weight loss secret that doctors don't want you to know! 
        Lose 50 pounds in just ONE WEEK with this SIMPLE trick! Pharmaceutical companies are trying 
        to HIDE this from you! Click NOW to discover the SECRET that will change your life FOREVER! 
        Limited time offer! Act now before Big Pharma takes this down! You won't believe what happens 
        next! This miracle cure will SHOCK you!!!""",
        source="clickbait-central.com",
        description="Example 4: Fake News - Obvious Clickbait"
    )
    
    # Example 5: Real news about science
    analyze_article(
        detector,
        title="New Study Published on Climate Change Effects",
        text="""Researchers at a major university published findings in the journal Nature this week 
        examining the effects of climate change on ocean temperatures. The peer-reviewed study analyzed 
        data collected over two decades from monitoring stations across the Atlantic Ocean. Lead 
        researcher Dr. Sarah Johnson noted that the findings align with previous climate models. 
        The research team emphasized the importance of continued monitoring and international 
        cooperation to address environmental challenges.""",
        source="nytimes.com",
        description="Example 5: Real News - Scientific Study + Credible Source"
    )
    
    # Example 6: Fake news with political bias
    analyze_article(
        detector,
        title="EXPOSED: Secret Plot to Control Everything!",
        text="""Leaked documents reveal massive conspiracy by global elites to control world events! 
        Insiders claim shadow government pulling all the strings behind the scenes! Wake up sheeple! 
        The mainstream media won't report this TRUTH! They're all in on it together! Share this 
        everywhere before they silence us! The deep state doesn't want you to know! Patriots must 
        unite against this tyranny! Trust no one! Do your own research!""",
        source="truth-seekers.blog",
        description="Example 6: Fake News - Conspiracy Theory"
    )
    
    # Example 7: Real news - sports
    analyze_article(
        detector,
        title="Local Team Wins Championship Game",
        text="""The home team secured a decisive victory in yesterday's championship match with a 
        final score of 3-1. Team captain praised the players' dedication throughout the season. 
        The win marks the franchise's first championship in five years. Fans celebrated in the 
        streets following the final whistle. Coach Martinez credited the team's defensive strategy 
        and strong performance from the goalkeeper. Season ticket sales are expected to increase 
        for next year.""",
        source="espn.com",
        description="Example 7: Real News - Sports Reporting"
    )
    
    # Example 8: Neutral/borderline content
    analyze_article(
        detector,
        title="Local Business Opens New Location",
        text="""A popular coffee shop opened its third location downtown this morning. The new store 
        features expanded seating and a full menu of beverages and pastries. Owner Maria Garcia 
        said she's excited to serve the growing neighborhood. The business started as a small 
        family operation five years ago. Regular customers attended the grand opening celebration. 
        The shop will be open daily from 6 AM to 8 PM.""",
        source="local-news-daily.com",
        description="Example 8: Real News - Local Community News"
    )
    
    # Summary
    print("\n" + "=" * 80)
    print(" 📊 TEST COMPLETE")
    print("=" * 80)
    print("\nThe fake news detector analyzed 8 different examples:")
    print("  • Real news from reliable sources (Reuters, BBC, NY Times)")
    print("  • Fake news with sensational language and unreliable sources")
    print("  • Clickbait content")
    print("  • Conspiracy theories")
    print("  • Local and sports news")
    print("\n💡 Key Detection Factors:")
    print("  1. Source reliability (reliable vs unreliable domains)")
    print("  2. Writing style (professional vs sensational)")
    print("  3. Language patterns (factual vs clickbait)")
    print("  4. Text characteristics (measured by Markov chains and TF-IDF)")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()

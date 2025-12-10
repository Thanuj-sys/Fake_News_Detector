# Web verification based on source reliability
import re

# Extended list of reliable sources with their scores
reliable_sources = {
    'nytimes': 0.95, 'reuters': 0.95, 'bbc': 0.93, 'apnews': 0.98, 
    'associatedpress': 0.98, 'theguardian': 0.90, 'guardian': 0.90,
    'washingtonpost': 0.92, 'wsj': 0.94, 'economist': 0.91,
    'npr': 0.90, 'pbs': 0.89, 'time': 0.85, 'nationalgeographic': 0.88,
    'bloomberg': 0.92, 'forbes': 0.83, 'ft': 0.93, 'financialtimes': 0.93,
    'newyorker': 0.87, 'atlantic': 0.86, 'politico': 0.82, 'axios': 0.81,
    'cbsnews': 0.85, 'abcnews': 0.84, 'nbcnews': 0.85, 'usatoday': 0.80,
    'cnn': 0.81, 'globaltimes': 0.70, 'dailynews': 0.75
}

# List of known unreliable/fake source keywords
unreliable_keywords = [
    'fake', 'clickbait', 'conspiracy', 'gossip', 'tabloid', 'rumor',
    'hoax', 'propaganda', 'satire', 'parody', 'buzz', 'viral',
    'shocking', 'secrets', 'truth', 'exposed', 'daily.fake'
]


def _canonicalize(s: str) -> str:
    if not s:
        return ''
    s = s.strip().lower()
    # if it's a URL, extract host
    if '//' in s and '.' in s:
        try:
            # remove scheme
            parts = s.split('//', 1)[1]
            host = parts.split('/')[0]
            s = host
        except Exception:
            pass
    # remove common tlds but keep the structure
    s = s.replace('.com', '').replace('.co.uk', '').replace('.uk', '')
    s = s.replace('.org', '').replace('.net', '').replace('.io', '')
    s = s.replace('-', '').replace(' ', '').replace('www.', '')
    return s


def verify_web(source):
    """Return a reliability score in [0.0, 1.0].
    Known reliable sources -> 0.80-0.98
    Unreliable keywords -> 0.0-0.10  
    Unknown sources -> 0.10-0.20 (assume untrusted)
    """
    if not source:
        return 0.15  # Low score for empty/unknown source
    
    s = _canonicalize(source)
    
    # Check against reliable sources (exact match or substring)
    for reliable, score in reliable_sources.items():
        if reliable == s or reliable in s or s in reliable:
            return score
    
    # Check against unreliable keywords
    source_lower = source.lower()
    for keyword in unreliable_keywords:
        if keyword in source_lower:
            return 0.0  # Very unreliable
    
    # Check for .fake TLD or similar suspicious patterns
    if '.fake' in source_lower or 'fake' in s:
        return 0.0
    
    # Unknown source - assume untrusted (low score)
    # This is safer than neutral 0.5
    return 0.15

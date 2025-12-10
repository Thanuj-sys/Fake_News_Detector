import unittest
try:
    import pandas as pd
except Exception:
    pd = None
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import preprocess_dataset
from src.detector import FakeNewsDetector

class TestFakeNewsDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load and preprocess dataset
        cls.df = preprocess_dataset('data/fake_news_dataset.csv')
        cls.detector = FakeNewsDetector()
        cls.detector.train(cls.df)

    def test_predict_real_and_fake(self):
        # Test prediction on a real and a fake sample
        real_sample = self.df[self.df['label'] == 'real'].iloc[0]
        fake_sample = self.df[self.df['label'] == 'fake'].iloc[0]

        real_pred = self.detector.predict(real_sample['title'], real_sample['clean_text'], real_sample['source'])
        fake_pred = self.detector.predict(fake_sample['title'], fake_sample['clean_text'], fake_sample['source'])

        self.assertFalse(real_pred, "Real news predicted as fake")
        self.assertTrue(fake_pred, "Fake news predicted as real")

if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the TextProcessor class.
"""

import os
import sys
import unittest

# Add the parent directory to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import text_processor

class TestTextProcessor(unittest.TestCase):
    """Tests for the TextProcessor class."""
    
    def setUp(self):
        """Set up the test case."""
        self.processor = text_processor.TextProcessor(
            language='en',
            remove_stopwords=True,
            remove_punctuation=True,
            lemmatize=True,
            lowercase=True
        )
        
        self.sample_text = """
        Text mining, also known as text data mining, is the process of transforming unstructured text into a structured format.
        It helps organizations find valuable insights from their data.
        """
    
    def test_preprocess_text(self):
        """Test the preprocess_text method."""
        preprocessed = self.processor.preprocess_text(self.sample_text)
        
        # Check that the text is lowercase
        self.assertEqual(preprocessed, preprocessed.lower())
        
        # Check that punctuation is removed
        self.assertNotIn(',', preprocessed)
        self.assertNotIn('.', preprocessed)
        
        # Check that the text is not empty
        self.assertTrue(len(preprocessed) > 0)
    
    def test_tokenize(self):
        """Test the tokenize method."""
        tokens = self.processor.tokenize(self.sample_text)
        
        # Check that tokens are returned
        self.assertTrue(len(tokens) > 0)
        
        # Check that stopwords are removed
        self.assertNotIn('is', tokens)
        self.assertNotIn('the', tokens)
        
        # Check that important words are kept
        self.assertIn('text', tokens)
        self.assertIn('mining', tokens)
        self.assertIn('data', tokens)
    
    def test_process(self):
        """Test the process method."""
        result = self.processor.process(self.sample_text)
        
        # Check that the result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that the result contains the expected keys
        self.assertIn('original', result)
        self.assertIn('preprocessed', result)
        self.assertIn('tokens', result)
        self.assertIn('sentences', result)
        
        # Check that the original text is preserved
        self.assertEqual(result['original'], self.sample_text)
        
        # Check that tokens are returned
        self.assertTrue(len(result['tokens']) > 0)
        
        # Check that sentences are returned
        self.assertTrue(len(result['sentences']) > 0)

if __name__ == '__main__':
    unittest.main() 
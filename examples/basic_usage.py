#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Usage Example
------------------
This script demonstrates how to use the text mining tool with minimal dependencies.
It shows how to preprocess text, tokenize it, and perform basic analysis.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import text_processor
from src.utils import file_handler

def main():
    # Sample text
    sample_text = """
    Text mining, also known as text data mining, is the process of transforming unstructured text into a structured format
    to identify meaningful patterns and new insights. By applying advanced analytical techniques, text mining helps
    organizations find valuable business intelligence that might otherwise remain undiscovered within their unstructured data.
    
    Text mining uses natural language processing (NLP), machine learning, and statistical methods to extract and analyze
    information from text sources. These sources can include emails, social media posts, customer reviews, survey responses,
    articles, and other documents. The goal is to discover hidden patterns, trends, and relationships that can inform
    decision-making and strategy.
    """
    
    print("=== Text Mining Tool - Basic Usage Example ===\n")
    
    # Create a text processor
    print("Creating text processor...")
    processor = text_processor.TextProcessor(
        language='en',
        remove_stopwords=True,
        remove_punctuation=True,
        lemmatize=True,
        lowercase=True
    )
    
    # Process the text
    print("Processing text...")
    result = processor.process(sample_text)
    
    # Print the results
    print("\n=== Results ===\n")
    print(f"Number of tokens: {len(result['tokens'])}")
    print(f"Number of sentences: {len(result['sentences'])}")
    
    print("\nPreprocessed text:")
    print(result['preprocessed'])
    
    print("\nTokens (first 20):")
    print(result['tokens'][:20])
    
    print("\nSentences:")
    for i, sentence in enumerate(result['sentences']):
        print(f"  {i+1}. {sentence}")
    
    # Save the results
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "basic_usage_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 
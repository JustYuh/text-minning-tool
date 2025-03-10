#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Usage Example
---------------------
This script demonstrates how to use the text mining tool with additional dependencies.
It shows how to perform advanced analyses such as sentiment analysis, named entity recognition,
and topic modeling.

Note: This example requires additional dependencies to be installed:
- spaCy: pip install spacy
- transformers: pip install transformers
- gensim: pip install gensim
- scikit-learn: pip install scikit-learn
- matplotlib: pip install matplotlib
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import optional dependencies
try:
    from src.preprocessing import text_processor
    from src.utils import file_handler
    
    # Import analysis modules
    try:
        from src.analysis import sentiment, entity_recognition, topic_modeling, keyword_extractor
        ANALYSIS_MODULES_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Some analysis modules could not be imported: {e}")
        print("Advanced analysis features will be disabled.")
        ANALYSIS_MODULES_AVAILABLE = False
    
    # Import visualization module
    try:
        import matplotlib.pyplot as plt
        from src.visualization import visualizer
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        print("Warning: Visualization modules could not be imported.")
        print("Visualization features will be disabled.")
        VISUALIZATION_AVAILABLE = False
except ImportError as e:
    print(f"Error: {e}")
    print("Please make sure you have installed the required dependencies.")
    sys.exit(1)

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
    
    Sentiment analysis is one of the most common applications of text mining. It involves determining the emotional tone
    or attitude expressed in text, often categorized as positive, negative, or neutral. Companies use sentiment analysis
    to understand customer opinions about their products or services.
    
    Named Entity Recognition (NER) is another important text mining technique. It identifies and classifies named entities
    in text into predefined categories like person names, organizations, locations, etc. This helps in extracting structured
    information from unstructured text.
    """
    
    print("=== Text Mining Tool - Advanced Usage Example ===\n")
    
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
    processed_text = processor.process(sample_text)
    
    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the processed text
    with open(output_dir / "advanced_processed_text.json", 'w', encoding='utf-8') as f:
        json.dump(processed_text, f, indent=2, ensure_ascii=False)
    
    # Perform advanced analyses if available
    if ANALYSIS_MODULES_AVAILABLE:
        results = {}
        
        # Sentiment Analysis
        try:
            print("\nPerforming sentiment analysis...")
            sentiment_analyzer = sentiment.SentimentAnalyzer(model_size='small')
            results['sentiment'] = sentiment_analyzer.analyze(sample_text)
            print(f"Sentiment: {results['sentiment']['label']} (Score: {results['sentiment']['score']:.2f})")
        except Exception as e:
            print(f"Error performing sentiment analysis: {e}")
        
        # Named Entity Recognition
        try:
            print("\nPerforming named entity recognition...")
            ner = entity_recognition.NamedEntityRecognizer()
            results['entities'] = ner.extract_entities(sample_text)
            print("Entities found:")
            for entity in results['entities']:
                print(f"  - {entity['text']} ({entity['label']})")
        except Exception as e:
            print(f"Error performing named entity recognition: {e}")
        
        # Keyword Extraction
        try:
            print("\nPerforming keyword extraction...")
            extractor = keyword_extractor.KeywordExtractor()
            results['keywords'] = extractor.extract_keywords(sample_text)
            print("Keywords:")
            for keyword in results['keywords'][:10]:
                print(f"  - {keyword['text']} (Score: {keyword['score']:.2f})")
        except Exception as e:
            print(f"Error performing keyword extraction: {e}")
        
        # Topic Modeling
        try:
            print("\nPerforming topic modeling...")
            topic_modeler = topic_modeling.TopicModeler(n_topics=3)
            results['topics'] = topic_modeler.extract_topics([sample_text])
            print("Topics:")
            for i, topic in enumerate(results['topics']):
                print(f"  Topic {i+1}: {', '.join(word for word, _ in topic[:5])}")
        except Exception as e:
            print(f"Error performing topic modeling: {e}")
        
        # Save the results
        with open(output_dir / "advanced_analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Visualization
        if VISUALIZATION_AVAILABLE:
            try:
                print("\nGenerating visualizations...")
                
                # Word Cloud
                visualizer.generate_wordcloud(processed_text['tokens'], output_dir / "wordcloud.png")
                
                # Sentiment Distribution
                if 'sentiment' in results:
                    labels = ['Negative', 'Neutral', 'Positive']
                    values = [0.1, 0.3, 0.6]  # Example values
                    plt.figure(figsize=(8, 6))
                    plt.bar(labels, values, color=['red', 'gray', 'green'])
                    plt.title('Sentiment Distribution')
                    plt.ylabel('Score')
                    plt.savefig(output_dir / "sentiment_distribution.png")
                
                print(f"Visualizations saved to {output_dir}")
            except Exception as e:
                print(f"Error generating visualizations: {e}")
    else:
        print("\nAdvanced analysis modules are not available.")
        print("Please install the required dependencies to enable advanced features.")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main() 
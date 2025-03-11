#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Mining Tool - Command Line Interface
This module provides a command-line interface for the Text Mining Tool.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from preprocessing.text_processor import TextProcessor
    from utils.file_handler import load_text_file, save_results
except ImportError:
    # Add the parent directory to the path so we can import from src
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.preprocessing.text_processor import TextProcessor
    from src.utils.file_handler import load_text_file, save_results

# Try to import optional modules
try:
    from analysis.sentiment import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    
try:
    from analysis.entities import NamedEntityRecognizer
    NER_AVAILABLE = True
except ImportError:
    NER_AVAILABLE = False
    
try:
    from analysis.keywords import KeywordExtractor
    KEYWORDS_AVAILABLE = True
except ImportError:
    KEYWORDS_AVAILABLE = False
    
try:
    from analysis.topics import TopicModeler
    TOPICS_AVAILABLE = True
except ImportError:
    TOPICS_AVAILABLE = False
    
try:
    from analysis.summarizer import TextSummarizer
    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    
try:
    from visualization.visualizer import generate_wordcloud, generate_sentiment_chart
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Text Mining Tool')
    
    # Input/output options
    parser.add_argument('--input', '-i', required=True, help='Input file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--format', '-f', default='txt', help='Input file format (txt, csv, json, pdf, docx, html)')
    
    # Preprocessing options
    parser.add_argument('--language', '-l', default='en', help='Language code (ISO 639-1)')
    parser.add_argument('--no-stopwords', action='store_true', help='Do not remove stopwords')
    parser.add_argument('--no-punctuation', action='store_true', help='Do not remove punctuation')
    parser.add_argument('--no-lemmatize', action='store_true', help='Do not lemmatize tokens')
    parser.add_argument('--stem', action='store_true', help='Stem tokens')
    parser.add_argument('--no-lowercase', action='store_true', help='Do not convert text to lowercase')
    parser.add_argument('--min-token-length', type=int, default=2, help='Minimum length of tokens to keep')
    parser.add_argument('--custom-stopwords', help='File containing custom stopwords (one per line)')
    
    # Analysis options
    parser.add_argument('--analysis', '-a', help='Comma-separated list of analyses to perform (sentiment, ner, keywords, topics, summary)')
    
    # Visualization options
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualizations')
    
    return parser.parse_args()


def process_file(file_path, args, text_processor, analyzers):
    """Process a single file."""
    print(f"Processing {file_path}...")
    
    # Load the file
    text = load_text_file(file_path)
    
    # Process the text
    result = text_processor.process(text)
    
    # Perform additional analyses if requested
    if args.analysis:
        analyses = args.analysis.split(',')
        
        if 'sentiment' in analyses and SENTIMENT_AVAILABLE and 'sentiment_analyzer' in analyzers:
            print("Performing sentiment analysis...")
            result['sentiment'] = analyzers['sentiment_analyzer'].analyze(text)
            
        if 'ner' in analyses and NER_AVAILABLE and 'ner' in analyzers:
            print("Performing named entity recognition...")
            result['entities'] = analyzers['ner'].extract_entities(text)
            
        if 'keywords' in analyses and KEYWORDS_AVAILABLE and 'keyword_extractor' in analyzers:
            print("Performing keyword extraction...")
            result['keywords'] = analyzers['keyword_extractor'].extract_keywords(text)
            
        if 'topics' in analyses and TOPICS_AVAILABLE and 'topic_modeler' in analyzers:
            print("Performing topic modeling...")
            result['topics'] = analyzers['topic_modeler'].extract_topics([text])
            
        if 'summary' in analyses and SUMMARIZER_AVAILABLE and 'text_summarizer' in analyzers:
            print("Performing text summarization...")
            result['summary'] = analyzers['text_summarizer'].summarize(text)
    
    # Generate visualizations if requested
    if args.visualize and VISUALIZATION_AVAILABLE:
        print("Generating visualizations...")
        
        # Create a directory for visualizations
        vis_dir = os.path.join(args.output, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate word cloud
        wordcloud_path = os.path.join(vis_dir, f"{os.path.basename(file_path)}_wordcloud.png")
        generate_wordcloud(result['tokens'], wordcloud_path)
        result['wordcloud'] = wordcloud_path
        
        # Generate sentiment chart if sentiment analysis was performed
        if 'sentiment' in result:
            sentiment_chart_path = os.path.join(vis_dir, f"{os.path.basename(file_path)}_sentiment.png")
            generate_sentiment_chart(result['sentiment'], sentiment_chart_path)
            result['sentiment_chart'] = sentiment_chart_path
    
    # Save the results
    output_file = os.path.join(args.output, f"{os.path.basename(file_path)}_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")
    
    return result


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load custom stopwords if provided
    custom_stopwords = None
    if args.custom_stopwords:
        with open(args.custom_stopwords, 'r', encoding='utf-8') as f:
            custom_stopwords = [line.strip() for line in f]
    
    # Create a text processor
    text_processor = TextProcessor(
        language=args.language,
        remove_stopwords=not args.no_stopwords,
        remove_punctuation=not args.no_punctuation,
        remove_numbers=False,
        lemmatize=not args.no_lemmatize,
        stem=args.stem,
        lowercase=not args.no_lowercase,
        min_token_length=args.min_token_length,
        custom_stopwords=custom_stopwords
    )
    
    # Create analyzers if needed
    analyzers = {}
    
    if args.analysis:
        analyses = args.analysis.split(',')
        
        if 'sentiment' in analyses and SENTIMENT_AVAILABLE:
            from analysis.sentiment import SentimentAnalyzer
            analyzers['sentiment_analyzer'] = SentimentAnalyzer()
            
        if 'ner' in analyses and NER_AVAILABLE:
            from analysis.entities import NamedEntityRecognizer
            analyzers['ner'] = NamedEntityRecognizer()
            
        if 'keywords' in analyses and KEYWORDS_AVAILABLE:
            from analysis.keywords import KeywordExtractor
            analyzers['keyword_extractor'] = KeywordExtractor()
            
        if 'topics' in analyses and TOPICS_AVAILABLE:
            from analysis.topics import TopicModeler
            analyzers['topic_modeler'] = TopicModeler()
            
        if 'summary' in analyses and SUMMARIZER_AVAILABLE:
            from analysis.summarizer import TextSummarizer
            analyzers['text_summarizer'] = TextSummarizer()
    
    # Process input
    start_time = time.time()
    
    if os.path.isfile(args.input):
        # Process a single file
        process_file(args.input, args, text_processor, analyzers)
    elif os.path.isdir(args.input):
        # Process all files in the directory with the specified format
        for file_name in os.listdir(args.input):
            if file_name.endswith(f".{args.format}"):
                file_path = os.path.join(args.input, file_name)
                process_file(file_path, args, text_processor, analyzers)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main() 
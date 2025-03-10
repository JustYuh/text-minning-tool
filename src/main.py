#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Text Mining Tool - Command Line Interface
-------------------------------------------------
This module serves as the entry point for the command-line interface of the text mining tool.
It parses command-line arguments and orchestrates the text mining pipeline.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import text_processor
from src.analysis import sentiment, entity_recognition, topic_modeling, text_classifier, summarizer, keyword_extractor
from src.utils import file_handler, logger_config

# Configure logging
logger = logger_config.setup_logger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Advanced Text Mining Tool - Extract insights from text data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', '-i', required=True, 
                        help='Path to input file or directory')
    parser.add_argument('--output', '-o', default='results',
                        help='Path to output directory')
    parser.add_argument('--analysis', '-a', default='all',
                        help='Analysis types to perform (comma-separated): sentiment,ner,topics,classify,summarize,keywords,all')
    parser.add_argument('--format', '-f', default='auto',
                        help='Input format: auto, txt, csv, pdf, docx, html, json')
    parser.add_argument('--lang', '-l', default='en',
                        help='Language of the text (ISO 639-1 code)')
    parser.add_argument('--model', '-m', default='default',
                        help='Model to use for analysis (default, small, large)')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Batch size for processing')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of worker processes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode')
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate command line arguments."""
    # Check if input path exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input}")
        return False
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    if not output_path.exists():
        logger.info(f"Creating output directory: {args.output}")
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate analysis types
    valid_analyses = {'sentiment', 'ner', 'topics', 'classify', 'summarize', 'keywords', 'all'}
    requested_analyses = set(args.analysis.split(','))
    if not requested_analyses.issubset(valid_analyses):
        invalid = requested_analyses - valid_analyses
        logger.error(f"Invalid analysis types: {', '.join(invalid)}")
        return False
    
    return True

def run_analysis(args):
    """Run the requested analysis on the input data."""
    logger.info(f"Starting analysis with parameters: {args}")
    
    # Load and preprocess the data
    try:
        documents = file_handler.load_documents(args.input, format=args.format)
        logger.info(f"Loaded {len(documents)} documents from {args.input}")
        
        # Preprocess the text
        processor = text_processor.TextProcessor(language=args.lang)
        processed_docs = processor.process_batch(documents, batch_size=args.batch_size, n_workers=args.workers)
        logger.info("Text preprocessing completed")
        
        # Determine which analyses to run
        analyses = args.analysis.split(',')
        run_all = 'all' in analyses
        
        results = {}
        
        # Run sentiment analysis
        if run_all or 'sentiment' in analyses:
            logger.info("Running sentiment analysis...")
            sentiment_analyzer = sentiment.SentimentAnalyzer(model_size=args.model)
            results['sentiment'] = sentiment_analyzer.analyze_batch(processed_docs)
        
        # Run named entity recognition
        if run_all or 'ner' in analyses:
            logger.info("Running named entity recognition...")
            ner = entity_recognition.NamedEntityRecognizer(model_size=args.model)
            results['entities'] = ner.extract_entities_batch(processed_docs)
        
        # Run topic modeling
        if run_all or 'topics' in analyses:
            logger.info("Running topic modeling...")
            topic_modeler = topic_modeling.TopicModeler(n_topics=10)
            results['topics'] = topic_modeler.extract_topics(processed_docs)
        
        # Run text classification
        if run_all or 'classify' in analyses:
            logger.info("Running text classification...")
            classifier = text_classifier.TextClassifier(model_size=args.model)
            results['categories'] = classifier.classify_batch(processed_docs)
        
        # Run text summarization
        if run_all or 'summarize' in analyses:
            logger.info("Running text summarization...")
            text_summarizer = summarizer.TextSummarizer(model_size=args.model)
            results['summaries'] = text_summarizer.summarize_batch(processed_docs)
        
        # Run keyword extraction
        if run_all or 'keywords' in analyses:
            logger.info("Running keyword extraction...")
            extractor = keyword_extractor.KeywordExtractor()
            results['keywords'] = extractor.extract_keywords_batch(processed_docs)
        
        # Save results
        output_path = Path(args.output)
        file_handler.save_results(results, output_path)
        logger.info(f"Analysis completed. Results saved to {args.output}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=args.debug)
        return False

def main():
    """Main entry point for the command-line interface."""
    args = parse_arguments()
    
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Run the analysis
    success = run_analysis(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 
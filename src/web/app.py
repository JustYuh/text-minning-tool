#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web interface for the Text Mining Tool.
This module provides a Flask-based web interface for the Text Mining Tool.
"""

import os
import json
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from flask import Flask, render_template, request, jsonify, send_file
    from werkzeug.utils import secure_filename
except ImportError:
    print("Flask is not installed. Please install it with: pip install flask")
    print("Continuing with limited functionality...")

try:
    from src.preprocessing.text_processor import TextProcessor
    from src.utils.file_handler import load_text_file, save_results
except ImportError:
    print("Could not import required modules. Make sure you're running from the project root.")
    sys.exit(1)

# Try to import analysis modules
try:
    from src.analysis.sentiment import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Sentiment analysis module not available.")

try:
    from src.analysis.entities import NamedEntityRecognizer
    NER_AVAILABLE = True
except ImportError:
    NER_AVAILABLE = False
    print("Named entity recognition module not available.")

try:
    from src.analysis.keywords import KeywordExtractor
    KEYWORDS_AVAILABLE = True
except ImportError:
    KEYWORDS_AVAILABLE = False
    print("Keyword extraction module not available.")

# Try to import visualization module
VISUALIZATION_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False
WORDCLOUD_AVAILABLE = False
PLOTLY_AVAILABLE = False
generate_wordcloud = None
generate_sentiment_chart = None
generate_entity_chart = None
generate_keyword_chart = None

try:
    from src.visualization import (
        generate_wordcloud,
        generate_sentiment_chart,
        generate_entity_chart,
        generate_keyword_chart,
        MATPLOTLIB_AVAILABLE,
        WORDCLOUD_AVAILABLE,
        PLOTLY_AVAILABLE
    )
    VISUALIZATION_AVAILABLE = any([MATPLOTLIB_AVAILABLE, WORDCLOUD_AVAILABLE, PLOTLY_AVAILABLE])
except ImportError:
    print("Visualization module not available.")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'text-mining-tool-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Set up upload and results folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize text processor and analyzers
text_processor = TextProcessor()
sentiment_analyzer = SentimentAnalyzer() if SENTIMENT_AVAILABLE else None
entity_recognizer = NamedEntityRecognizer() if NER_AVAILABLE else None
keyword_extractor = KeywordExtractor() if KEYWORDS_AVAILABLE else None

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', features={
        'sentiment': SENTIMENT_AVAILABLE,
        'ner': NER_AVAILABLE,
        'keywords': KEYWORDS_AVAILABLE,
        'visualization': VISUALIZATION_AVAILABLE,
        'matplotlib': MATPLOTLIB_AVAILABLE,
        'wordcloud': WORDCLOUD_AVAILABLE,
        'plotly': PLOTLY_AVAILABLE
    })

@app.route('/process', methods=['POST'])
def process():
    """Process the submitted text and return results."""
    text = request.form.get('text', '')
    if not text:
        return render_template('index.html', error="No text provided")
    
    # Process the text
    result = {'original_text': text}
    processed = text_processor.process(text)
    result.update(processed)
    
    # Perform analyses
    if SENTIMENT_AVAILABLE and sentiment_analyzer:
        result['sentiment'] = sentiment_analyzer.analyze(text)
        
        if MATPLOTLIB_AVAILABLE:
            sentiment_chart_path = 'static/results/sentiment_chart.png'
            full_path = os.path.join(app.static_folder, 'results', 'sentiment_chart.png')
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if generate_sentiment_chart(result['sentiment'], full_path):
                result['sentiment_chart'] = sentiment_chart_path
    
    if NER_AVAILABLE and entity_recognizer:
        result['entities'] = entity_recognizer.extract_entities(text)
        
        if MATPLOTLIB_AVAILABLE:
            entity_chart_path = 'static/results/entity_chart.png'
            full_path = os.path.join(app.static_folder, 'results', 'entity_chart.png')
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if generate_entity_chart(result['entities'], full_path):
                result['entity_chart'] = entity_chart_path
    
    if KEYWORDS_AVAILABLE and keyword_extractor:
        result['keywords'] = keyword_extractor.extract_keywords(text)
        
        if WORDCLOUD_AVAILABLE:
            wordcloud_path = 'static/results/wordcloud.png'
            full_path = os.path.join(app.static_folder, 'results', 'wordcloud.png')
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if generate_wordcloud(result['tokens'], full_path):
                result['wordcloud'] = wordcloud_path
        
        if MATPLOTLIB_AVAILABLE:
            keyword_chart_path = 'static/results/keyword_chart.png'
            full_path = os.path.join(app.static_folder, 'results', 'keyword_chart.png')
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if generate_keyword_chart(result['keywords'], full_path):
                result['keyword_chart'] = keyword_chart_path
    
    return render_template('results.html', result=result)

@app.route('/api/process', methods=['POST'])
def api_process():
    """API endpoint for processing text."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    result = {'original_text': text}
    
    # Process the text
    processed = text_processor.process(text)
    result.update(processed)
    
    # Perform analyses based on what's available
    if SENTIMENT_AVAILABLE and sentiment_analyzer:
        result['sentiment'] = sentiment_analyzer.analyze(text)
    
    if NER_AVAILABLE and entity_recognizer:
        result['entities'] = entity_recognizer.extract_entities(text)
    
    if KEYWORDS_AVAILABLE and keyword_extractor:
        result['keywords'] = keyword_extractor.extract_keywords(text)
    
    return jsonify(result)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analyzing text."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    analysis_type = data.get('analysis_type', 'all')
    result = {}
    
    if analysis_type == 'sentiment' and SENTIMENT_AVAILABLE and sentiment_analyzer:
        result['sentiment'] = sentiment_analyzer.analyze(text)
    elif analysis_type == 'entities' and NER_AVAILABLE and entity_recognizer:
        result['entities'] = entity_recognizer.extract_entities(text)
    elif analysis_type == 'keywords' and KEYWORDS_AVAILABLE and keyword_extractor:
        result['keywords'] = keyword_extractor.extract_keywords(text)
    elif analysis_type == 'all':
        if SENTIMENT_AVAILABLE and sentiment_analyzer:
            result['sentiment'] = sentiment_analyzer.analyze(text)
        if NER_AVAILABLE and entity_recognizer:
            result['entities'] = entity_recognizer.extract_entities(text)
        if KEYWORDS_AVAILABLE and keyword_extractor:
            result['keywords'] = keyword_extractor.extract_keywords(text)
    else:
        return jsonify({"error": f"Analysis type '{analysis_type}' not supported"}), 400
    
    return jsonify(result)

@app.route('/api/visualize', methods=['POST'])
def api_visualize():
    """API endpoint for generating visualizations."""
    if not VISUALIZATION_AVAILABLE:
        return jsonify({"error": "Visualization module not available"}), 501
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    # Process the text
    processed = text_processor.process(data['text'])
    
    # Create a unique filename
    import time
    timestamp = int(time.time())
    visualization_type = data.get('visualization_type', 'wordcloud')
    
    if visualization_type == 'wordcloud' and WORDCLOUD_AVAILABLE:
        if processed['tokens']:
            wordcloud_path = os.path.join('results', f'wordcloud_{timestamp}.png')
            full_path = os.path.join(app.config['RESULTS_FOLDER'], f'wordcloud_{timestamp}.png')
            if generate_wordcloud(processed['tokens'], full_path):
                return jsonify({"visualization_url": f"/static/{wordcloud_path}"})
    
    elif visualization_type == 'sentiment' and MATPLOTLIB_AVAILABLE and SENTIMENT_AVAILABLE:
        sentiment = sentiment_analyzer.analyze(data['text'])
        sentiment_chart_path = os.path.join('results', f'sentiment_{timestamp}.png')
        full_path = os.path.join(app.config['RESULTS_FOLDER'], f'sentiment_{timestamp}.png')
        if generate_sentiment_chart(sentiment, full_path):
            return jsonify({"visualization_url": f"/static/{sentiment_chart_path}"})
    
    return jsonify({"error": f"Visualization type '{visualization_type}' not supported"}), 400

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("\nStarting Text Mining Tool Web Interface...")
    print("Available features:")
    print(f"- Sentiment Analysis: {'Available' if SENTIMENT_AVAILABLE else 'Not Available'}")
    print(f"- Named Entity Recognition: {'Available' if NER_AVAILABLE else 'Not Available'}")
    print(f"- Keyword Extraction: {'Available' if KEYWORDS_AVAILABLE else 'Not Available'}")
    print(f"- Visualization: {'Available' if VISUALIZATION_AVAILABLE else 'Not Available'}")
    if VISUALIZATION_AVAILABLE:
        print("  - Matplotlib:", "Available" if MATPLOTLIB_AVAILABLE else "Not Available")
        print("  - WordCloud:", "Available" if WORDCLOUD_AVAILABLE else "Not Available")
        print("  - Plotly:", "Available" if PLOTLY_AVAILABLE else "Not Available")
    print("\nOpen your browser and navigate to http://localhost:5000")
    app.run(host='0.0.0.0', debug=True) 
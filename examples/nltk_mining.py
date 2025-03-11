#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple text mining script using NLTK.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import string
import matplotlib.pyplot as plt
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text by tokenizing, removing stopwords, and stemming.
    """
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize into words
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return {
        'sentences': sentences,
        'tokens': tokens,
        'stemmed_tokens': stemmed_tokens,
        'lemmatized_tokens': lemmatized_tokens
    }

def analyze_text(processed_text):
    """
    Analyze the processed text.
    """
    # Count word frequencies
    fdist = FreqDist(processed_text['tokens'])
    
    # Find bigrams
    bigram_finder = BigramCollocationFinder.from_words(processed_text['tokens'])
    bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)
    
    # Calculate sentence length statistics
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in processed_text['sentences']]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    
    return {
        'word_frequencies': fdist,
        'bigrams': bigrams,
        'sentence_lengths': sentence_lengths,
        'avg_sentence_length': avg_sentence_length
    }

def visualize_results(analysis_results):
    """
    Visualize the analysis results.
    """
    # Plot word frequencies
    plt.figure(figsize=(12, 6))
    analysis_results['word_frequencies'].plot(30, cumulative=False)
    plt.title('Top 30 Word Frequencies')
    plt.savefig('word_frequencies.png')
    
    # Plot sentence length distribution
    plt.figure(figsize=(12, 6))
    plt.hist(analysis_results['sentence_lengths'], bins=20)
    plt.title('Sentence Length Distribution')
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Frequency')
    plt.savefig('sentence_lengths.png')
    
    print("Visualizations saved as 'word_frequencies.png' and 'sentence_lengths.png'")

def main():
    # Sample text
    text = """
    Text mining, also known as text data mining, is the process of transforming unstructured text into a structured format
    to identify meaningful patterns and new insights. By applying advanced analytical techniques, text mining helps
    organizations find valuable business intelligence that might otherwise remain undiscovered within their unstructured data.
    
    Text mining uses natural language processing (NLP), machine learning, and statistical methods to extract and analyze
    information from text sources. These sources can include emails, social media posts, customer reviews, survey responses,
    articles, and other documents. The goal is to discover hidden patterns, trends, and relationships that can inform
    decision-making and strategy.
    
    Common text mining tasks include:
    
    1. Information Extraction: Identifying and extracting specific pieces of information from text, such as names, dates,
       locations, and events.
    
    2. Text Categorization: Automatically assigning predefined categories or tags to text documents based on their content.
    
    3. Sentiment Analysis: Determining the emotional tone or attitude expressed in text, often categorized as positive,
       negative, or neutral.
    
    4. Topic Modeling: Discovering abstract topics or themes that occur in a collection of documents.
    
    5. Named Entity Recognition: Identifying and classifying named entities in text into predefined categories like person
       names, organizations, locations, etc.
    
    6. Relationship Extraction: Identifying relationships between entities mentioned in text.
    
    Text mining has applications across various industries, including healthcare, finance, marketing, customer service,
    and research. For example, healthcare organizations might use text mining to analyze patient records and identify
    trends in symptoms or treatment outcomes. Marketing teams might analyze social media posts to understand customer
    sentiment about their products or services.
    
    As the volume of unstructured text data continues to grow exponentially, text mining becomes increasingly valuable
    for organizations seeking to gain insights and competitive advantages from their data assets.
    """
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Analyze text
    analysis_results = analyze_text(processed_text)
    
    # Print results
    print("Text Analysis Results:")
    print(f"Number of sentences: {len(processed_text['sentences'])}")
    print(f"Number of words (after preprocessing): {len(processed_text['tokens'])}")
    print(f"Average sentence length: {analysis_results['avg_sentence_length']:.2f} words")
    
    print("\nTop 10 most frequent words:")
    for word, freq in analysis_results['word_frequencies'].most_common(10):
        print(f"{word}: {freq}")
    
    print("\nTop 10 bigrams:")
    for bigram in analysis_results['bigrams']:
        print(f"{bigram[0]} {bigram[1]}")
    
    # Visualize results
    try:
        visualize_results(analysis_results)
    except Exception as e:
        print(f"Could not create visualizations: {e}")

if __name__ == "__main__":
    main() 
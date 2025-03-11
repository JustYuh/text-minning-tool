# Text Mining Tool API Documentation

This document provides information about the API of the Text Mining Tool.

## Preprocessing Module

### TextProcessor Class

The `TextProcessor` class provides methods for preprocessing and normalizing text data.

```python
from src.preprocessing import text_processor

# Create a text processor
processor = text_processor.TextProcessor(
    language='en',
    remove_stopwords=True,
    remove_punctuation=True,
    remove_numbers=False,
    lemmatize=True,
    stem=False,
    lowercase=True,
    min_token_length=2,
    custom_stopwords=None,
    use_spacy=True
)

# Process a text
result = processor.process("Your text here")

# Process a batch of texts
results = processor.process_batch(["Text 1", "Text 2", "Text 3"])
```

#### Parameters

- `language` (str): The language code (ISO 639-1) for the text. Default: 'en'
- `remove_stopwords` (bool): Whether to remove stopwords. Default: True
- `remove_punctuation` (bool): Whether to remove punctuation. Default: True
- `remove_numbers` (bool): Whether to remove numbers. Default: False
- `lemmatize` (bool): Whether to lemmatize tokens. Default: True
- `stem` (bool): Whether to stem tokens. Default: False
- `lowercase` (bool): Whether to convert text to lowercase. Default: True
- `min_token_length` (int): Minimum length of tokens to keep. Default: 2
- `custom_stopwords` (List[str]): Additional stopwords to remove. Default: None
- `use_spacy` (bool): Whether to use spaCy for advanced NLP (if available). Default: True

#### Methods

- `preprocess_text(text: str) -> str`: Preprocess the text by applying various cleaning operations.
- `tokenize(text: str) -> List[str]`: Tokenize the text into words.
- `process(text: str) -> Dict[str, Any]`: Process the text and return various representations.
- `process_batch(texts: List[str], batch_size: int = 64, n_workers: int = 4) -> List[Dict[str, Any]]`: Process a batch of texts in parallel.

## Analysis Module

### SentimentAnalyzer Class

The `SentimentAnalyzer` class provides methods for analyzing sentiment in text data.

```python
from src.analysis import sentiment

# Create a sentiment analyzer
analyzer = sentiment.SentimentAnalyzer(model_size='default')

# Analyze a text
result = analyzer.analyze("Your text here")

# Analyze a batch of texts
results = analyzer.analyze_batch(["Text 1", "Text 2", "Text 3"])
```

### NamedEntityRecognizer Class

The `NamedEntityRecognizer` class provides methods for extracting named entities from text data.

```python
from src.analysis import entity_recognition

# Create a named entity recognizer
ner = entity_recognition.NamedEntityRecognizer()

# Extract entities from a text
entities = ner.extract_entities("Your text here")

# Extract entities from a batch of texts
entities_batch = ner.extract_entities_batch(["Text 1", "Text 2", "Text 3"])
```

### TopicModeler Class

The `TopicModeler` class provides methods for extracting topics from a collection of documents.

```python
from src.analysis import topic_modeling

# Create a topic modeler
topic_modeler = topic_modeling.TopicModeler(n_topics=10)

# Extract topics from a collection of documents
topics = topic_modeler.extract_topics(["Document 1", "Document 2", "Document 3"])
```

### TextClassifier Class

The `TextClassifier` class provides methods for classifying text into predefined categories.

```python
from src.analysis import text_classifier

# Create a text classifier
classifier = text_classifier.TextClassifier()

# Classify a text
result = classifier.classify("Your text here")

# Classify a batch of texts
results = classifier.classify_batch(["Text 1", "Text 2", "Text 3"])
```

### TextSummarizer Class

The `TextSummarizer` class provides methods for generating summaries of lengthy documents.

```python
from src.analysis import summarizer

# Create a text summarizer
text_summarizer = summarizer.TextSummarizer()

# Summarize a text
summary = text_summarizer.summarize("Your text here")

# Summarize a batch of texts
summaries = text_summarizer.summarize_batch(["Text 1", "Text 2", "Text 3"])
```

### KeywordExtractor Class

The `KeywordExtractor` class provides methods for extracting important terms and phrases from text data.

```python
from src.analysis import keyword_extractor

# Create a keyword extractor
extractor = keyword_extractor.KeywordExtractor()

# Extract keywords from a text
keywords = extractor.extract_keywords("Your text here")

# Extract keywords from a batch of texts
keywords_batch = extractor.extract_keywords_batch(["Text 1", "Text 2", "Text 3"])
```

## Utils Module

### File Handler

The `file_handler` module provides utilities for loading and saving files in various formats.

```python
from src.utils import file_handler

# Load a text file
text = file_handler.load_text_file("path/to/file.txt")

# Load documents from a directory
documents = file_handler.load_documents("path/to/directory", format="txt")

# Save results to a directory
file_handler.save_results(results, "path/to/output/directory")
```

## Visualization Module

The `visualizer` module provides utilities for visualizing text mining results.

```python
from src.visualization import visualizer

# Generate a word cloud
visualizer.generate_wordcloud(tokens, "path/to/output/wordcloud.png")

# Generate a sentiment distribution chart
visualizer.generate_sentiment_chart(sentiment_results, "path/to/output/sentiment.png")
``` 
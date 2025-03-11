#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Named Entity Recognizer Module
This module provides the NamedEntityRecognizer class for extracting named entities from text data.
"""

import re
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    
    # Download required NLTK data if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
        
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker')
        
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
        
    NLTK_NER_AVAILABLE = True
except ImportError:
    NLTK_NER_AVAILABLE = False
    print("NLTK NER is not available. Basic entity recognition will be used.")


class NamedEntityRecognizer:
    """
    A class for extracting named entities from text data.
    
    This class provides methods for identifying and extracting named entities
    (such as people, organizations, locations) from text data using NLTK's
    named entity recognition capabilities.
    
    Attributes:
        use_nltk (bool): Whether to use NLTK's named entity recognition.
    """
    
    def __init__(self, use_nltk: bool = True):
        """
        Initialize the NamedEntityRecognizer.
        
        Args:
            use_nltk: Whether to use NLTK's named entity recognition (if available).
        """
        self.use_nltk = use_nltk and NLTK_NER_AVAILABLE
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from the text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A list of dictionaries containing the extracted entities.
        """
        if not text:
            return []
        
        if self.use_nltk:
            return self._extract_entities_with_nltk(text)
        else:
            return self._extract_entities_with_regex(text)
    
    def _extract_entities_with_nltk(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from the text using NLTK.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A list of dictionaries containing the extracted entities.
        """
        entities = []
        
        try:
            # Tokenize the text
            tokens = word_tokenize(text)
            
            # Perform part-of-speech tagging
            pos_tags = pos_tag(tokens, lang='eng')
            
            # Perform named entity recognition
            named_entities = ne_chunk(pos_tags)
            
            # Extract entities from the parse tree
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    # This is a named entity
                    entity_type = chunk.label()
                    entity_text = ' '.join(c[0] for c in chunk)
                    
                    entities.append({
                        'text': entity_text,
                        'type': entity_type,
                        'start': text.find(entity_text),
                        'end': text.find(entity_text) + len(entity_text)
                    })
        except Exception as e:
            print(f"Error in NLTK entity extraction: {str(e)}")
            # Fallback to regex-based extraction
            return self._extract_entities_with_regex(text)
        
        return entities
    
    def _extract_entities_with_regex(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from the text using regular expressions.
        
        This is a very basic implementation that uses regular expressions to
        identify potential named entities. It's not as accurate as NLTK's
        named entity recognition, but it can be used as a fallback.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A list of dictionaries containing the extracted entities.
        """
        entities = []
        
        # Extract potential person names (capitalized words)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        for match in re.finditer(person_pattern, text):
            entity_text = match.group(0)
            entities.append({
                'text': entity_text,
                'type': 'PERSON',
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract potential organization names (all caps words)
        org_pattern = r'\b[A-Z]{2,}\b'
        for match in re.finditer(org_pattern, text):
            entity_text = match.group(0)
            entities.append({
                'text': entity_text,
                'type': 'ORGANIZATION',
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract potential locations (capitalized words followed by common location words)
        loc_pattern = r'\b[A-Z][a-z]+ (?:Street|Avenue|Road|Boulevard|Lane|Drive|Place|Square|Court|Park|Bridge|River|Ocean|Sea|Mountain|Valley|Forest|Desert|City|Town|Village|County|State|Province|Country|Island)\b'
        for match in re.finditer(loc_pattern, text):
            entity_text = match.group(0)
            entities.append({
                'text': entity_text,
                'type': 'LOCATION',
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def extract_entities_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract named entities from a batch of texts.
        
        Args:
            texts: The texts to analyze.
            
        Returns:
            A list of lists of dictionaries containing the extracted entities.
        """
        return [self.extract_entities(text) for text in texts] 
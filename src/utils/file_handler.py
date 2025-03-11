#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Handler Module
This module provides utilities for loading and saving files in various formats.
"""

import os
import json
import csv
from typing import Dict, List, Any, Optional, Union

def load_text_file(file_path: str) -> str:
    """
    Load a text file.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        The contents of the file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with a different encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        The contents of the file as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv_file(file_path: str, delimiter: str = ',') -> List[Dict[str, str]]:
    """
    Load a CSV file.
    
    Args:
        file_path: The path to the file.
        delimiter: The delimiter used in the CSV file.
        
    Returns:
        The contents of the file as a list of dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)

def load_pdf_file(file_path: str) -> str:
    """
    Load a PDF file.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        The contents of the file as a string.
    """
    try:
        from PyPDF2 import PdfReader
        
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        return text
    except ImportError:
        raise ImportError("PyPDF2 is required to load PDF files. Install it with: pip install PyPDF2")

def load_docx_file(file_path: str) -> str:
    """
    Load a Word document.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        The contents of the file as a string.
    """
    try:
        import docx
        
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text
    except ImportError:
        raise ImportError("python-docx is required to load Word documents. Install it with: pip install python-docx")

def load_html_file(file_path: str) -> str:
    """
    Load an HTML file.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        The contents of the file as a string.
    """
    try:
        from bs4 import BeautifulSoup
        
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text()
    except ImportError:
        raise ImportError("beautifulsoup4 is required to load HTML files. Install it with: pip install beautifulsoup4")

def load_file(file_path: str, file_format: Optional[str] = None) -> Union[str, Dict[str, Any], List[Dict[str, str]]]:
    """
    Load a file based on its format.
    
    Args:
        file_path: The path to the file.
        file_format: The format of the file. If None, the format is inferred from the file extension.
        
    Returns:
        The contents of the file.
    """
    if file_format is None:
        file_format = os.path.splitext(file_path)[1][1:].lower()
    
    if file_format in ['txt', 'text']:
        return load_text_file(file_path)
    elif file_format in ['json']:
        return load_json_file(file_path)
    elif file_format in ['csv']:
        return load_csv_file(file_path)
    elif file_format in ['pdf']:
        return load_pdf_file(file_path)
    elif file_format in ['docx', 'doc']:
        return load_docx_file(file_path)
    elif file_format in ['html', 'htm']:
        return load_html_file(file_path)
    else:
        # Default to text file
        return load_text_file(file_path)

def load_documents(directory_path: str, file_format: str = 'txt') -> List[str]:
    """
    Load all documents in a directory with the specified format.
    
    Args:
        directory_path: The path to the directory.
        file_format: The format of the files to load.
        
    Returns:
        A list of document contents.
    """
    documents = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(f".{file_format}"):
            file_path = os.path.join(directory_path, file_name)
            documents.append(load_file(file_path, file_format))
    
    return documents

def save_text_file(text: str, file_path: str) -> None:
    """
    Save text to a file.
    
    Args:
        text: The text to save.
        file_path: The path to the file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save.
        file_path: The path to the file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_csv_file(data: List[Dict[str, Any]], file_path: str, delimiter: str = ',') -> None:
    """
    Save data to a CSV file.
    
    Args:
        data: The data to save.
        file_path: The path to the file.
        delimiter: The delimiter to use in the CSV file.
    """
    if not data:
        return
    
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)

def save_results(results: Dict[str, Any], output_dir: str, base_name: str = 'results') -> None:
    """
    Save results to various file formats.
    
    Args:
        results: The results to save.
        output_dir: The directory to save the results to.
        base_name: The base name for the output files.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    save_json_file(results, json_path)
    
    # Save tokens and sentences as text
    if 'tokens' in results:
        tokens_path = os.path.join(output_dir, f"{base_name}_tokens.txt")
        save_text_file(' '.join(results['tokens']), tokens_path)
    
    if 'sentences' in results:
        sentences_path = os.path.join(output_dir, f"{base_name}_sentences.txt")
        save_text_file('\n'.join(results['sentences']), sentences_path)
    
    # Save summary if available
    if 'summary' in results:
        summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
        save_text_file(results['summary'], summary_path)
    
    # Save as CSV (flattened)
    csv_data = []
    for key, value in results.items():
        if isinstance(value, list):
            if all(isinstance(item, dict) for item in value):
                # List of dictionaries (e.g., entities, keywords)
                for item in value:
                    item_data = {'type': key}
                    item_data.update(item)
                    csv_data.append(item_data)
            else:
                # Simple list (e.g., tokens, sentences)
                csv_data.append({'type': key, 'value': ' '.join(str(item) for item in value)})
        elif isinstance(value, dict):
            # Dictionary (e.g., sentiment)
            item_data = {'type': key}
            item_data.update(value)
            csv_data.append(item_data)
        elif not isinstance(value, (str, int, float, bool)):
            # Skip complex types
            continue
        else:
            # Simple value
            csv_data.append({'type': key, 'value': value})
    
    if csv_data:
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        save_csv_file(csv_data, csv_path) 
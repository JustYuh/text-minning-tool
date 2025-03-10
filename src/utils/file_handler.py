#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Handler Module
-----------------
This module provides utilities for loading and saving files in various formats,
including text, CSV, PDF, DOCX, and JSON.
"""

import os
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, BinaryIO, TextIO

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)

def load_text_file(file_path: Union[str, Path]) -> str:
    """
    Load text from a plain text file.
    
    Args:
        file_path: The path to the text file.
        
    Returns:
        The text content of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"Error loading text file {file_path}: {str(e)}")
        return ""

def load_csv_file(file_path: Union[str, Path], text_column: Optional[str] = None) -> List[str]:
    """
    Load text from a CSV file.
    
    Args:
        file_path: The path to the CSV file.
        text_column: The name of the column containing the text data.
        
    Returns:
        A list of text strings from the specified column.
    """
    try:
        df = pd.read_csv(file_path)
        
        # If text_column is not specified, use the first column
        if text_column is None:
            text_column = df.columns[0]
        
        # Check if the column exists
        if text_column not in df.columns:
            logger.warning(f"Column '{text_column}' not found in CSV file. Using first column.")
            text_column = df.columns[0]
        
        # Extract text from the column
        texts = df[text_column].astype(str).tolist()
        
        return texts
    
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {str(e)}")
        return []

def load_excel_file(file_path: Union[str, Path], text_column: Optional[str] = None, sheet_name: Optional[str] = None) -> List[str]:
    """
    Load text from an Excel file.
    
    Args:
        file_path: The path to the Excel file.
        text_column: The name of the column containing the text data.
        sheet_name: The name of the sheet to load.
        
    Returns:
        A list of text strings from the specified column.
    """
    try:
        # Load the Excel file
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        # If text_column is not specified, use the first column
        if text_column is None:
            text_column = df.columns[0]
        
        # Check if the column exists
        if text_column not in df.columns:
            logger.warning(f"Column '{text_column}' not found in Excel file. Using first column.")
            text_column = df.columns[0]
        
        # Extract text from the column
        texts = df[text_column].astype(str).tolist()
        
        return texts
    
    except Exception as e:
        logger.error(f"Error loading Excel file {file_path}: {str(e)}")
        return []

def load_pdf_file(file_path: Union[str, Path]) -> str:
    """
    Load text from a PDF file.
    
    Args:
        file_path: The path to the PDF file.
        
    Returns:
        The text content of the PDF.
    """
    try:
        import PyPDF2
        
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            return text
    
    except ImportError:
        logger.error("PyPDF2 is not installed. Install it with 'pip install PyPDF2'.")
        return ""
    except Exception as e:
        logger.error(f"Error loading PDF file {file_path}: {str(e)}")
        return ""

def load_docx_file(file_path: Union[str, Path]) -> str:
    """
    Load text from a DOCX file.
    
    Args:
        file_path: The path to the DOCX file.
        
    Returns:
        The text content of the DOCX.
    """
    try:
        import docx
        
        doc = docx.Document(file_path)
        text = ""
        
        # Extract text from paragraphs
        for para in doc.paragraphs:
            text += para.text + "\n"
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"
        
        return text
    
    except ImportError:
        logger.error("python-docx is not installed. Install it with 'pip install python-docx'.")
        return ""
    except Exception as e:
        logger.error(f"Error loading DOCX file {file_path}: {str(e)}")
        return ""

def load_json_file(file_path: Union[str, Path], text_field: Optional[str] = None) -> Union[List[str], Dict[str, Any]]:
    """
    Load text from a JSON file.
    
    Args:
        file_path: The path to the JSON file.
        text_field: The field containing the text data.
        
    Returns:
        A list of text strings or the JSON data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If text_field is specified, extract text from that field
        if text_field:
            if isinstance(data, list):
                texts = []
                for item in data:
                    if isinstance(item, dict) and text_field in item:
                        texts.append(str(item[text_field]))
                return texts
            elif isinstance(data, dict) and text_field in data:
                return [str(data[text_field])]
        
        # Otherwise, return the raw JSON data
        return data
    
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return []

def load_html_file(file_path: Union[str, Path]) -> str:
    """
    Load text from an HTML file.
    
    Args:
        file_path: The path to the HTML file.
        
    Returns:
        The text content of the HTML.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        # Parse HTML and extract text
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    except Exception as e:
        logger.error(f"Error loading HTML file {file_path}: {str(e)}")
        return ""

def load_url(url: str) -> str:
    """
    Load text from a URL.
    
    Args:
        url: The URL to load.
        
    Returns:
        The text content of the URL.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        # Parse HTML and extract text
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    except Exception as e:
        logger.error(f"Error loading URL {url}: {str(e)}")
        return ""

def load_document(file_path: Union[str, Path]) -> str:
    """
    Load text from a document, automatically detecting the file type.
    
    Args:
        file_path: The path to the document.
        
    Returns:
        The text content of the document.
    """
    file_path = Path(file_path)
    
    # Check if the file exists
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return ""
    
    # Determine file type based on extension
    extension = file_path.suffix.lower()
    
    if extension == '.txt':
        return load_text_file(file_path)
    elif extension == '.csv':
        texts = load_csv_file(file_path)
        return '\n'.join(texts)
    elif extension in ['.xls', '.xlsx']:
        texts = load_excel_file(file_path)
        return '\n'.join(texts)
    elif extension == '.pdf':
        return load_pdf_file(file_path)
    elif extension == '.docx':
        return load_docx_file(file_path)
    elif extension == '.json':
        data = load_json_file(file_path)
        if isinstance(data, list):
            return '\n'.join(str(item) for item in data)
        else:
            return str(data)
    elif extension in ['.html', '.htm']:
        return load_html_file(file_path)
    else:
        logger.warning(f"Unsupported file type: {extension}. Trying as text file.")
        return load_text_file(file_path)

def load_documents(input_path: Union[str, Path], format: str = 'auto') -> List[str]:
    """
    Load documents from a file or directory.
    
    Args:
        input_path: The path to the file or directory.
        format: The format of the input ('auto', 'txt', 'csv', 'pdf', 'docx', 'json', 'html').
        
    Returns:
        A list of document texts.
    """
    input_path = Path(input_path)
    
    # Check if the path exists
    if not input_path.exists():
        logger.error(f"Path not found: {input_path}")
        return []
    
    # If the path is a directory, load all files in it
    if input_path.is_dir():
        documents = []
        
        # Get all files in the directory
        files = list(input_path.glob('*'))
        
        # Filter files by format if specified
        if format != 'auto':
            files = [f for f in files if f.suffix.lower() == f'.{format}']
        
        # Load each file
        for file_path in files:
            if file_path.is_file():
                text = load_document(file_path)
                if text:
                    documents.append(text)
        
        return documents
    
    # If the path is a file, load it
    elif input_path.is_file():
        text = load_document(input_path)
        return [text] if text else []
    
    # If the path is neither a file nor a directory, it might be a URL
    elif str(input_path).startswith(('http://', 'https://')):
        text = load_url(str(input_path))
        return [text] if text else []
    
    else:
        logger.error(f"Invalid path: {input_path}")
        return []

def save_text_file(text: str, file_path: Union[str, Path]) -> bool:
    """
    Save text to a plain text file.
    
    Args:
        text: The text to save.
        file_path: The path to the output file.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        logger.error(f"Error saving text file {file_path}: {str(e)}")
        return False

def save_csv_file(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> bool:
    """
    Save data to a CSV file.
    
    Args:
        data: The data to save.
        file_path: The path to the output file.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        logger.error(f"Error saving CSV file {file_path}: {str(e)}")
        return False

def save_json_file(data: Union[List[Any], Dict[str, Any]], file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save.
        file_path: The path to the output file.
        indent: The indentation level for the JSON file.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        # Convert numpy arrays and other non-serializable objects to lists
        def convert_to_serializable(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            elif isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.to_dict()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            else:
                return obj
        
        # Recursively convert objects in lists and dictionaries
        def convert_container(obj):
            if isinstance(obj, dict):
                return {k: convert_container(convert_to_serializable(v)) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_container(convert_to_serializable(item)) for item in obj]
            else:
                return convert_to_serializable(obj)
        
        serializable_data = convert_container(data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=indent, ensure_ascii=False)
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        return False

def save_results(results: Dict[str, Any], output_dir: Union[str, Path]) -> bool:
    """
    Save analysis results to files.
    
    Args:
        results: The analysis results to save.
        output_dir: The directory to save the results to.
        
    Returns:
        True if successful, False otherwise.
    """
    output_dir = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the full results as JSON
    full_results_path = output_dir / 'results.json'
    if not save_json_file(results, full_results_path):
        return False
    
    # Save individual result types to separate files
    for result_type, result_data in results.items():
        # Skip the original and processed text
        if result_type in ['original_text', 'processed_text']:
            continue
        
        # Save as JSON
        result_path = output_dir / f'{result_type}.json'
        if not save_json_file(result_data, result_path):
            logger.warning(f"Failed to save {result_type} results to {result_path}")
        
        # Save as CSV if the data is a list of dictionaries
        if isinstance(result_data, list) and all(isinstance(item, dict) for item in result_data):
            csv_path = output_dir / f'{result_type}.csv'
            if not save_csv_file(result_data, csv_path):
                logger.warning(f"Failed to save {result_type} results to {csv_path}")
    
    logger.info(f"Results saved to {output_dir}")
    return True 
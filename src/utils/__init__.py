"""
Utils Module
This module provides utility functions for the Text Mining Tool.
"""

from .file_handler import (
    load_text_file,
    load_json_file,
    load_csv_file,
    load_pdf_file,
    load_docx_file,
    load_html_file,
    load_file,
    load_documents,
    save_text_file,
    save_json_file,
    save_csv_file,
    save_results
)

__all__ = [
    'load_text_file',
    'load_json_file',
    'load_csv_file',
    'load_pdf_file',
    'load_docx_file',
    'load_html_file',
    'load_file',
    'load_documents',
    'save_text_file',
    'save_json_file',
    'save_csv_file',
    'save_results'
] 
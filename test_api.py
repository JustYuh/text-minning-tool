#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Text Mining Tool API
This script tests the API endpoints of the Text Mining Tool
"""

import requests
import json
import sys

def test_process_endpoint():
    """Test the /api/process endpoint"""
    print("Testing /api/process endpoint...")
    
    url = "http://localhost:5000/api/process"
    data = {
        "text": "This is a test sentence. The Text Mining Tool is working properly. This is great!"
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        print("Status: Success")
        print(f"Tokens: {len(result['tokens'])} tokens found")
        print(f"Sentences: {len(result['sentences'])} sentences found")
        print("Sample tokens:", result['tokens'][:5])
        print("Sample sentences:", result['sentences'])
        print()
        return True
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False

def test_analyze_endpoint():
    """Test the /api/analyze endpoint"""
    print("Testing /api/analyze endpoint...")
    
    url = "http://localhost:5000/api/analyze"
    data = {
        "text": "I love using this Text Mining Tool. It's amazing and works perfectly!",
        "analyses": ["sentiment", "entities", "keywords"]
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        print("Status: Success")
        
        if "sentiment" in result:
            print(f"Sentiment: {result['sentiment']['label']} (score: {result['sentiment']['score']})")
        
        if "entities" in result:
            print(f"Entities: {len(result['entities'])} entities found")
            for entity in result['entities']:
                print(f"  - {entity['text']} ({entity['type']})")
        
        if "keywords" in result:
            print(f"Keywords: {len(result['keywords'])} keywords found")
            for keyword in result['keywords']:
                print(f"  - {keyword['text']} (score: {keyword['score']})")
        
        print()
        return True
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False

def main():
    """Main function"""
    print("Text Mining Tool API Test\n")
    
    process_success = test_process_endpoint()
    analyze_success = test_analyze_endpoint()
    
    if process_success and analyze_success:
        print("All tests passed! The API is working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
# Tests for the Text Mining Tool

This directory contains tests for the Text Mining Tool.

## Running Tests

To run all tests:

```bash
python -m unittest discover tests
```

To run a specific test file:

```bash
python -m unittest tests/test_text_processor.py
```

To run a specific test case:

```bash
python -m unittest tests.test_text_processor.TestTextProcessor
```

To run a specific test method:

```bash
python -m unittest tests.test_text_processor.TestTextProcessor.test_preprocess_text
```

## Test Files

- `test_text_processor.py`: Tests for the TextProcessor class

## Writing Tests

When writing tests, follow these guidelines:

1. Create a new test file for each module or class you want to test
2. Use the `unittest` framework
3. Name test files with the prefix `test_`
4. Name test methods with the prefix `test_`
5. Use descriptive method names that indicate what is being tested
6. Include assertions that verify the expected behavior
7. Use the `setUp` method to set up test fixtures
8. Use the `tearDown` method to clean up after tests 
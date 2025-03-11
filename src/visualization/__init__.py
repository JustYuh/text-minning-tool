"""
Visualization Module
This module provides functions for visualizing text mining results.
"""

# Initialize availability flags
MATPLOTLIB_AVAILABLE = False
WORDCLOUD_AVAILABLE = False
PLOTLY_AVAILABLE = False
SEABORN_AVAILABLE = False

# Initialize function placeholders
generate_wordcloud = lambda *args, **kwargs: False
generate_sentiment_chart = lambda *args, **kwargs: False
generate_entity_chart = lambda *args, **kwargs: False
generate_keyword_chart = lambda *args, **kwargs: False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Matplotlib not available. Some visualizations will be disabled.")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    print("WordCloud not available. Word cloud generation will be disabled.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Plotly not available. Interactive visualizations will be disabled.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("Seaborn not available. Some visualizations will be disabled.")

if all([MATPLOTLIB_AVAILABLE, SEABORN_AVAILABLE]):
    try:
        from .visualizer import (
            generate_wordcloud,
            generate_sentiment_chart,
            generate_entity_chart,
            generate_keyword_chart
        )
        print("Successfully imported visualization functions")
    except ImportError as e:
        print(f"Error importing visualization functions: {str(e)}")
        print("Using fallback visualization functions that return False")

# Export all symbols
__all__ = [
    'generate_wordcloud',
    'generate_sentiment_chart',
    'generate_entity_chart',
    'generate_keyword_chart',
    'MATPLOTLIB_AVAILABLE',
    'WORDCLOUD_AVAILABLE',
    'PLOTLY_AVAILABLE',
    'SEABORN_AVAILABLE'
] 
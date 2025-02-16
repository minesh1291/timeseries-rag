Pattern Recognition
==================

This guide demonstrates the pattern recognition capabilities of Time Series RAG.

Pattern Extraction
----------------

.. literalinclude:: scripts/pattern_recognition.py
   :language: python
   :lines: 1-7
   :caption: Pattern Recognition Examples

Extract and analyze patterns in time series data:

.. literalinclude:: scripts/pattern_recognition.py
   :language: python
   :pyobject: extract_patterns_example

Pattern Search
-------------

Search for similar patterns using the RAG system:

.. literalinclude:: scripts/pattern_recognition.py
   :language: python
   :pyobject: pattern_search_example

Seasonality Analysis
------------------

Analyze seasonal patterns in time series:

.. literalinclude:: scripts/pattern_recognition.py
   :language: python
   :pyobject: seasonality_analysis_example

Running the Examples
------------------

Execute all examples with:

.. code-block:: bash

   python pattern_recognition.py
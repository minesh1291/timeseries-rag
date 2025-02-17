Anomaly Detection
================

This guide demonstrates the anomaly detection capabilities of Time Series RAG.

Synthetic Data Generation
-----------------------

.. literalinclude:: scripts/anomaly_detection.py
   :language: python
   :lines: 1-7
   :caption: Anomaly Detection Examples

Create synthetic sensor data with anomalies:

.. literalinclude:: scripts/anomaly_detection.py
   :language: python
   :pyobject: create_synthetic_sensor_data

Anomaly Detection
---------------

Detect anomalies in sensor data:

.. literalinclude:: scripts/anomaly_detection.py
   :language: python
   :pyobject: detect_anomalies_example

Anomaly Pattern Analysis
----------------------

Analyze patterns in detected anomalies:

.. literalinclude:: scripts/anomaly_detection.py
   :language: python
   :pyobject: analyze_anomaly_patterns

Complete analysis example:

.. literalinclude:: scripts/anomaly_detection.py
   :language: python
   :pyobject: anomaly_pattern_analysis_example

Running the Examples
------------------

Execute all examples with:

.. code-block:: bash

   python anomaly_detection.py
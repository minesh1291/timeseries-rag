Usage Examples
=============

Basic Usage
----------

Here are some examples of how to use the Time Series RAG system:

Using the Python API
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from timeseries_rag.models import TimeSeriesEmbedder
    from timeseries_rag.rag import TimeSeriesRAG, TimeSeriesDocument
    import numpy as np

    # Initialize components
    embedder = TimeSeriesEmbedder(target_length=256)
    rag = TimeSeriesRAG()

    # Create some example time series
    t = np.linspace(0, 10, 100)
    sine_wave = np.sin(t)
    noisy_sine = sine_wave + np.random.normal(0, 0.2, size=len(sine_wave))

    # Embed and store the first time series
    embedding = embedder.embed(sine_wave)
    doc = TimeSeriesDocument(
        id="sine_1",
        data=sine_wave,
        metadata={"type": "sine", "frequency": 1.0},
        embedding=embedding
    )
    rag.add_document(doc)

    # Search for similar patterns
    query_embedding = embedder.embed(noisy_sine)
    results = rag.search(query_embedding, k=5)

    # Print results
    for result in results:
        print(f"Document ID: {result['id']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print()

Using the Web API
~~~~~~~~~~~~~~~

.. code-block:: python

    import requests
    import pandas as pd
    import numpy as np

    # Create example data
    t = np.linspace(0, 10, 100)
    sine_wave = np.sin(t)
    
    # Save to CSV
    pd.DataFrame(sine_wave).to_csv('sine.csv', index=False)

    # Upload time series
    files = {'file': open('sine.csv', 'rb')}
    metadata = '{"type": "sine", "frequency": 1.0}'
    response = requests.post(
        'http://localhost:50758/upload',
        files=files,
        data={'metadata': metadata}
    )
    print(response.json())

    # Search for similar patterns
    noisy_sine = sine_wave + np.random.normal(0, 0.2, size=len(sine_wave))
    pd.DataFrame(noisy_sine).to_csv('query.csv', index=False)

    files = {'file': open('query.csv', 'rb')}
    response = requests.post(
        'http://localhost:50758/search',
        files=files,
        params={'k': 5}
    )
    print(response.json())
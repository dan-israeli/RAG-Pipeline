# RAG Pipeline
Implementation of a RAG pipeline to assist an LLM in answering news-related queries.

The implementation includes the use and utiltzation of a vector database.

## Running Instructions
In order to run the provided code, please follow the steps below:

1 .Make sure you have the relevant packages installed using the following command:

```bash
pip install -U sentence-transformers datasets pinecone-client cohere tqdm
```

2. Place the ```RAG_pipeline.py``` in some directory on your machine. Then, create the following .txt files in the directory:
   - cohere_api_key.txt - contains your API key for thr Cohere LLM.
   - pinecone_api_key.txt - contains your API key for the Pinecone vector database service.

    Note that you can generate those API keys using the links below:

    - Cohere: https://cohere.com/

    - Pinecone: https://www.pinecone.io/

3. Run the main Python code file:

```bash
  python RAG_pipeline.py
```

**Remarks:**

1. The runtime for the procedure might be quite long. When running is finished, the outputs will be printed out.
   
2. the LLM's outputs might vary in different runs. Therefore, the answers to the queries might differ from the ones presented in the report. However, the outputs should be overall similar.

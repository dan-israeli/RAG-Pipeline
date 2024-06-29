# RAG Pipeline
Implementation of a RAG pipeline to assist an LLM in answering news-related queries.

The implementation includes the use and utiltzation of a vector database.

## Running Instructions
First, make sure you have the relevant packages installed using the following command:

```bash
pip install -U sentence-transformers datasets pinecone-client cohere tqdm
```

Second, place the ```RAG_pipeline.py``` in some directory on your machine. Then, place two .txt files into the same directory, named ```cohere_api_key.txt``` and ```pinecone_api_key```. They should contain your API keys for the Cohere LLM and the Pinecone service, respectively.

You can generate those API keys in the links below:

- Cohere: https://cohere.com/

- Pinecone: https://www.pinecone.io/

Lastly, run the main Python code file:

```bash
python RAG_pipeline.py
```

The runtime for the procedure might be quite long. When running is finished, outputs similar to the ones in the report will be printed out.

**Note:** the LLM's outputs might vary in different runs. Therefore the answers to the queries might differ from the ones presented in the report.

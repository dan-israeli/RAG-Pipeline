### imports

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import cohere
import numpy as np
import warnings

warnings.filterwarnings("ignore")


### connect to APIs and define constants

with open("cohere_api_key.txt") as f:
    COHERE_API_KEY = f.read().strip()
with open("pinecone_api_key.txt") as f:
    PINECONE_API_KEY = f.read().strip()

DATASET_NAME = 'vblagoje/cc_news'  # news articles dataset
REC_NUM = 2000
INDEX_NAME = 'hw1-index-team15'  # index for the Pinecone vectors database
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # BERT embedding model (dim=384)
LLM_NAME = 'command-r-plus'  # Cohere model


### define the functions provided in tutorial 3

def load_and_embedd_dataset(
        dataset_name: str = DATASET_NAME,
        split: str = 'train',
        model: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL),
        text_field: str = 'text',
        rec_num: int = REC_NUM) -> tuple:
    """
    Load a dataset and embedd the text field using a sentence-transformer model
    Args:
        dataset_name: The name of the dataset to load
        split: The split of the dataset to load
        model: The model to use for embedding
        text_field: The field in the dataset that contains the text
        rec_num: The number of records to load and embedd
    Returns:
        tuple: A tuple containing the dataset and the embeddings
    """

    print("Loading and embedding the dataset")

    # Load the dataset
    dataset = load_dataset(dataset_name, split=split).select(range(rec_num))

    # Embed the first `rec_num` rows of the dataset
    embeddings = model.encode(dataset[text_field][:rec_num])

    print("Done!")
    return dataset, embeddings


def create_pinecone_index(
        index_name: str,
        dimension: int,
        metric: str = 'cosine'):
    """
    Create a pinecone index if it does not exist
    Args:
        index_name: The name of the index
        dimension: The dimension of the index
        metric: The metric to use for the index
    Returns:
        Pinecone: A pinecone object which can later be used for upserting vectors and connecting to VectorDBs
    """
    print("Creating a Pinecone index...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            # Remember! It is crucial that the metric you will use in your VectorDB will also be a metric your embedding
            # model works well with!
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    print("Done!")
    return pc


def upsert_vectors(
        index: Pinecone,
        embeddings: np.ndarray,
        dataset: dict,
        text_field: str = 'text',
        batch_size: int = 128):
    """
    Upsert vectors to a pinecone index
    Args:
        index: The pinecone index object
        embeddings: The embeddings to upsert
        dataset: The dataset containing the metadata
        batch_size: The batch size to use for upserting
    Returns:
        An updated pinecone index
    """
    print("Upserting the embeddings to the Pinecone index...")
    shape = embeddings.shape

    ids = [str(i) for i in range(shape[0])]
    meta = [{text_field: text} for text in dataset[text_field]]

    # create list of (id, vector, metadata) tuples to be upserted
    to_upsert = list(zip(ids, embeddings, meta))

    for i in tqdm(range(0, shape[0], batch_size)):
        i_end = min(i + batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])
    return index


def augment_prompt(
        query: str,
        model: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL),
        index=None):
    """
    Augment the prompt with the top 3 results from the knowledge base
    Args:
        query: The query to augment
        index: The vectorstore object
    Returns:
        str: The augmented prompt
    """
    results = [float(val) for val in list(model.encode(query))]

    # get top 3 results from knowledge base
    query_results = index.query(
        vector=results,
        top_k=3,
        include_values=True,
        include_metadata=True
    )['matches']

    text_matches = [match['metadata']['text'] for match in query_results]

    # get the text from the results
    source_knowledge = "\n\n".join(text_matches)

    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.
    Contexts:
    {source_knowledge}
    If the answer is not included in the source knowledge - say that you don't know.
    Query: {query}"""
    return augmented_prompt, source_knowledge


### define our functions

def compare_query_results(query_lst, embedding_model, index):
    """
    Gets list of queries, an embedding model, and a vector DB index. Displays the queries as well as
    the LLM's output on them. We compare the output of the LLM with VS without receiving additional
    documents from the vector DB as context.
    """
    for query in query_lst:
        # initialize LLM
        co = cohere.Client(api_key=COHERE_API_KEY)

        # receive response
        response = co.chat(
            model=LLM_NAME,
            message=query)
        response = response.text

        # receive informed response
        augmented_prompt, source_knowledge = augment_prompt(query, model=embedding_model, index=index)
        response_informed = co.chat(
            model=LLM_NAME,
            message=augmented_prompt)
        response_informed = response_informed.text

        print(f"Q: {query}\n")
        print(f"A: {response}\n")
        print(f"A (informed): {response_informed}\n")
        print(f"Source Knowledge:\n{source_knowledge}")
        print("\n----------------------------------\n")


def main():
    """
    Main function
    """
    # first, load and embed the dataset
    model = SentenceTransformer(EMBEDDING_MODEL)

    dataset, embeddings = load_and_embedd_dataset(
        dataset_name=DATASET_NAME,
        rec_num=REC_NUM,
        model=model)

    shape = embeddings.shape

    # now, create the vector database
    pc = create_pinecone_index(INDEX_NAME, shape[1])

    # initialize the index and insert embeddings vectors
    index = pc.Index(INDEX_NAME)
    index = upsert_vectors(index, embeddings, dataset)

    # define the anecdotal queries and compare them
    query_lst = ["In the Brodway's revival of the Carousel show, who was cast to play Jigger Craigin?",
                 "Who was the fourth artistic director of NBT?",
                 "What was Chris Pratt's first commercial?"]

    compare_query_results(query_lst, model, index)


if __name__ == "__main__":
    main()

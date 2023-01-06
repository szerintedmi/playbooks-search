"""
Search using SentenceTransformers
"""

import torchUtils
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import time
from sentence_transformers import util
import numpy as np


RESULT_MIN_SCORE = 0.4  # don't inlcude content from result below this score
MODEL_NAME = "multi-qa-mpnet-base-dot-v1"  # 'nq-distilbert-base-v1'
# Number of passages we want to retrieve with the bi-encoder (16 is the max on mps )
TOP_K = 16


def search(query: str, top_k: int = TOP_K) -> list({int, float}):
    """
    Search the corpus with Sentenfor the query and return the top_k results
    returns an ordered list of dicts with keys: score, corpus_id
    """
    start_time = time.time()

    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)

    hits = util.semantic_search(
        question_embedding, corpus_embeddings, top_k=TOP_K)
    hits = hits[0]  # Get the hits for the first query

    end_time = time.time()

    # Output of top-k hits
    print("Input question:", query)
    print("Results in {:.3f} seconds.".format(end_time - start_time))
    return hits


@st.experimental_singleton
def get_corpus() -> pd.DataFrame:
    """ Returns the corpus loaded as dataframe """
    return df


torch_device = torchUtils.getDevice()

df = pd.read_parquet('corpus/embeddings_multi-qa-mpnet.parquet')
print("Corpus loaded from corpus/embeddings_multi-qa-mpnet.parquet")

bi_encoder = SentenceTransformer(MODEL_NAME, device=torch_device)

# load the embeddings from the dataframe to a torch tensor
corpus_embeddings = np.array(df["embeddings"].to_list(), dtype=np.float32)
corpus_embeddings = torch.from_numpy(
    corpus_embeddings).to(torch_device)

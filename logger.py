"""
Logging functions 
TODO: error handling
"""
import os
from supabase import create_client, Client

url: str = os.getenv("SUPABASE_PROJECT_URL")
key: str = os.getenv("SUPABASE_API_KEY")

supabase: Client = create_client(url, key)


def log_search(query: str, embedding_model: str, corpus_result: object, completion_params: object,
               completion_response: str, token_usage: int, completion_time: float) -> None:
    """
    Log the search query and the result to supadb database
    """
    data = supabase.table("query_logs").insert(
        {"query": query, "embedding_model": embedding_model,
         "corpus_result": corpus_result[["corpus_id", 'score', 'fullTitle', 'tokensLength']].to_dict("records"),
         "completion_params": completion_params, "completion_response": completion_response,
         "token_usage": token_usage, "completion_time": completion_time}).execute()

    return

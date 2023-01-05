"""
Logging functions 
TODO: error handling
"""
import os
from supabase import create_client, Client

url: str = os.getenv("SUPABASE_PROJECT_URL")
key: str = os.getenv("SUPABASE_API_KEY")

supabase: Client = create_client(url, key)


def log_search(query: str, embedding_model: str, corpus_result: object, question_prompt: str,
               pronpt_response: str, token_usage: int) -> None:
    """
    Log the search query and the result to supadb database
    """
    data = supabase.table("query_logs").insert(
        {"query": query, "embedding_model": embedding_model,
         "corpus_result": corpus_result[["corpus_id", 'score', 'fullTitle', 'tokensLength']].to_dict("records"),
         "question_prompt": question_prompt, "prompt_response": pronpt_response, "token_usage": token_usage}).execute()

    return

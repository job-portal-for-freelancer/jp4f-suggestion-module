from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from qdrant_client import QdrantClient, models

# Qdrant connection details
url_qdrant = r"https://a6ffce05-6187-4d07-908f-1f99523272b0.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key_qdrant = r"Njte5BIsSsLujzQKxUF3BfcUimvvH2V9GFEQ9FIcnk9NpCtqmwg1Lw"

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(url=url_qdrant, api_key=api_key_qdrant)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Qdrant cluster: {e}")

# Create FastAPI app
app = FastAPI()

# Vector padding function
def pad_vector(vector: List[float], vector_size: int) -> List[float]:
    return vector + [0] * (vector_size - len(vector))

# Request models
class QueryRequest(BaseModel):
    vector: List[float]

class BatchQueryRequest(BaseModel):
    vectors: List[QueryRequest]

class SingleQueryRequest(BaseModel):
    vector: List[float]

@app.post("/search")
def search_single(request: SingleQueryRequest):
    try:
        results = qdrant_client.search(
            collection_name="jp4f-vector",
            query_vector=pad_vector(request.vector, 100),
            limit=3,
        )
        return [
            {"id": point.id, "score": point.score, "payload": point.payload}
            for point in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")

# Batch query API
@app.post("/search_batch")
def search_batch(request: BatchQueryRequest):
    try:
        # Prepare search queries
        search_queries = [
            models.QueryRequest(query=pad_vector(query.vector, 100), limit=3)
            for query in request.vectors
        ]

        # Perform batch search
        search_results = qdrant_client.query_batch_points(
            collection_name="jp4f-vector",
            requests=search_queries,
        )

        # Format and return results
        formatted_results = [
            {
                "query": search_query.query,
                "results": [
                    {"id": point.id, "score": point.score, "payload": point.payload}
                    for point in search_result.points  # Access the points attribute here
                ],
            }
            for search_query, search_result in zip(search_queries, search_results)
        ]
        return {"results": formatted_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch search: {e}")

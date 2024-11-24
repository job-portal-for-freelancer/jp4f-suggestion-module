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
    """Pads the vector to match the required size."""
    if len(vector) < vector_size:
        return vector + [0] * (vector_size - len(vector))
    return vector[:vector_size]

# Request models
class QueryRequest(BaseModel):
    vector: List[float]

class BatchQueryRequest(BaseModel):
    vectors: List[QueryRequest]

class SingleQueryRequest(BaseModel):
    vector: List[float]

@app.post("/search")
async def search_single(request: SingleQueryRequest):
    try:
        # Perform the search query
        results = qdrant_client.search(
            collection_name="jp4f-vector",
            query_vector=pad_vector(request.vector, 100),
            limit=3,
        )
        # Format and return results
        return [
            {"id": point.id, "score": point.score, "payload": point.payload}
            for point in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")

@app.post("/search_batch")
async def search_batch(request: BatchQueryRequest):
    try:
        # Pre-compute padded vectors to minimize repetitive operations
        padded_vectors = [
            models.QueryRequest(query=pad_vector(query.vector, 100), limit=3)
            for query in request.vectors
        ]

        # Perform batch search
        search_results = qdrant_client.query_batch_points(
            collection_name="jp4f-vector",
            requests=padded_vectors,
        )

        # Format and return results
        formatted_results = [
            {
                "query": search_query.query,
                "results": [
                    {"id": point.id, "score": point.score, "payload": point.payload}
                    for point in search_result.points
                ],
            }
            for search_query, search_result in zip(padded_vectors, search_results)
        ]
        return {"results": formatted_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch search: {e}")

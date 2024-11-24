from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from qdrant_client import QdrantClient, models
import numpy as np
import asyncio

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

# Vector padding function (NumPy optimized)
def pad_vector(vector: List[float], vector_size: int) -> np.ndarray:
    """Pads or trims the vector to match the required size."""
    padded = np.zeros(vector_size, dtype=np.float32)
    padded[: len(vector)] = vector[:vector_size]
    return padded

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
        # Pre-process vector and perform search query
        query_vector = pad_vector(request.vector, 100)  # Match the vector size of the collection
        results = await asyncio.to_thread(
            qdrant_client.search,
            collection_name="jp4f-vector",
            query_vector=query_vector.tolist(),
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
        # Pre-compute padded vectors
        vector_size = 100  # Match the vector size of the collection
        padded_queries = [
            models.QueryRequest(
                query=pad_vector(query.vector, vector_size).tolist(),
                limit=3,
            )
            for query in request.vectors
        ]

        # Perform batch search
        search_results = await asyncio.to_thread(
            qdrant_client.query_batch_points,
            collection_name="jp4f-vector",
            requests=padded_queries,
        )

        # Format and return results
        formatted_results = [
            {
                "query": padded_query.query,
                "results": [
                    {"id": point.id, "score": point.score, "payload": point.payload}
                    for point in search_result.points
                ],
            }
            for padded_query, search_result in zip(padded_queries, search_results)
        ]
        return {"results": formatted_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch search: {e}")

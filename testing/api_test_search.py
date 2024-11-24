from qdrant_client import QdrantClient, models
from fastapi import FastAPI, HTTPException

# Define Qdrant connection details
url_qdrant = r"https://a6ffce05-6187-4d07-908f-1f99523272b0.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key_qdrant = r"Njte5BIsSsLujzQKxUF3BfcUimvvH2V9GFEQ9FIcnk9NpCtqmwg1Lw"

# Connect to Qdrant
try:
    qdrant_client = QdrantClient(url=url_qdrant, api_key=api_key_qdrant)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Qdrant cluster: {e}")



@app.post("/sp")
async def infer_texts(request: InferenceRequest):
result = qdrant_client.query_points(
    collection_name="HiSpeHiPre_Sparse",
    query=models.SparseVector(indices=[1, 3, 5, 7], values=[0.1, 0.2, 0.3, 0.4]),
    using="text",
).points

print("Result", result)


@app.post("/sparse")
async def infer_texts(request: InferenceRequest):
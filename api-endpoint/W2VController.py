from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# import tritonclient.grpc as grpcclient
import tritonclient.http as httplient

import numpy as np

import argparse
import uvicorn
from qdrant_client import QdrantClient, models
import random


app = FastAPI(
       docs_url="/python/docs",  # URL to access Swagger UI
    redoc_url="/python/redoc",  # URL to access ReDoc documentation
    openapi_url="/python/openapi.json"  # URL to access OpenAPI schema
)


MODEL_NAME = "bls_w2v"
MODEL_VERSION = "1"
TRITON_URL = "localhost:1234"


qdrant_client = QdrantClient(
    api_key="WbvUZU0wqIyJBwLHrbKI9mpiHRUCY1EAH--tdKdi2x0QX2tdoPoiTg",
    https="https://a6ffce05-6187-4d07-908f-1f99523272b0.us-east4-0.gcp.cloud.qdrant.io"
)

triton_client = httplient.InferenceServerClient(url=TRITON_URL, verbose=False)

class InputProjectDto(BaseModel):
    projectDescription: str

class InputResumseDto(BaseModel):
    resumeDescription: str


@app.post("/python/store-project")
async def vectorize_store(input_data: InputProjectDto):
    try:
        text_embeds = await w2c(input_data)
      
        _ = await store_qdrant(text_embeds)
        
        
        return {
            "EmbeddedProject": text_embeds.tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    

@app.post("/python/get-matching-projects")
async def infer_method1(input_data: InputResumseDto):
    try:
        text_embeds = await w2c(input_data)
      
        _ = await search_qdrant(text_embeds)
        
        return {
            "embeddings": text_embeds.tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")    


async def w2c(input_data: InputProjectDto) -> np.ndarray:
    try:
        text_tensor = np.array([input_data.projectDescription], dtype=np.object_)
        
        # text_input = grpcclient.InferInput("TEXT", text_tensor.shape, "BYTES")
        text_input = httplient.InferInput("TEXT", text_tensor.shape, "BYTES")
        
        
        text_input.set_data_from_numpy(text_tensor)

        # output = grpcclient.InferRequestedOutput("VEC")
        output = httplient.InferRequestedOutput("VEC")

        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=[text_input],
            outputs=[output],
            model_version=MODEL_VERSION
        )

        text_embeds = response.as_numpy("VEC")
        return text_embeds[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")   


async def store_qdrant(text_embeds: np.ndarray):
    #Store this vector to Qdrant
    random_id = random.randint(1, 1_000_000)

    # Store vector in Qdrant
    _ = qdrant_client.upsert(
        collection_name="jp4f-vector-db",
        points=[
            models.PointStruct(
                id=random_id,
                vector=text_embeds.tolist(),
                payload={"id": random_id},  # Add any other metadata as needed
            )
        ],
    )  

async def search_qdrant(text_embeds: np.ndarray):
    search_result = qdrant_client.query_points(
    collection_name="jp4f-vector-db",
    query=text_embeds.tolist(), 
    search_params=models.SearchParams(hnsw_ef=128, exact=False),
    limit=20,
    with_payload=True, 
    )
    print("Search result:", search_result)
    print(type(search_result))
    points = search_result.points

# Prepare the response by iterating through the points
    results = []
    for point in points:
        results.append({
            "id": point.id,
            "score": point.score,
            "payload": point.payload,
        })
        # Display search results
    for result in results:
        print(f"ID: {result['id']}, Score: {result['score']}, Payload: {result['payload']}")



def main():
    parser = argparse.ArgumentParser(description="Run FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on.")
    args = parser.parse_args()

    # Run the FastAPI app
    print(f"Starting FastAPI server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

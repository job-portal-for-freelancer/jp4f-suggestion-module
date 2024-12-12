from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tritonclient.grpc as grpcclient
# import tritonclient.http as httplient
import requests
import pyodbc
import uuid
import gc
import numpy as np
import argparse
import uvicorn
from qdrant_client import QdrantClient, models
import sys

# Prevent .pyc files
sys.dont_write_bytecode = True

# FastAPI application
app = FastAPI(
    docs_url="/python/docs",  # URL to access Swagger UI
    redoc_url="/python/redoc",  # URL to access ReDoc documentation
    openapi_url="/python/openapi.json"  # URL to access OpenAPI schema
)


# Database configuration
DB_CONFIG = {
    "DRIVER": "{ODBC Driver 17 for SQL Server}",
    "SERVER": "34.87.95.20,1433",
    "DATABASE": "JP4F",
    "UID": "sa",
    "PWD": "@dmin123",
}


MODEL_NAME = "bls_w2v"
MODEL_VERSION = "1"
TRITON_URL = "0.0.0.0:1235"
LOAD_URL = "0.0.0.0:1234"


qdrant_client = QdrantClient(
    api_key="WbvUZU0wqIyJBwLHrbKI9mpiHRUCY1EAH--tdKdi2x0QX2tdoPoiTg",
    url="https://a6ffce05-6187-4d07-908f-1f99523272b0.us-east4-0.gcp.cloud.qdrant.io"
)

triton_client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)


class TextInput(BaseModel):
    text: str

@app.post("/python/store-project")
async def vectorize_store():
    """
    Fetch projects from the database, vectorize their text, and store them in Qdrant.
    """
    try:
        query = """
            SELECT p.Id, p.JobTittle, p.JobDesciption, STRING_AGG(s.SkillName, ', ') AS Skills
            FROM Project p
            LEFT JOIN ProjectSkill ps ON ps.ProjectsId = p.Id
            LEFT JOIN Skill s ON ps.SkillsId = s.Id
            GROUP BY p.Id, p.JobTittle, p.JobDesciption;
        """
        with pyodbc.connect(
            f"DRIVER={DB_CONFIG['DRIVER']};SERVER={DB_CONFIG['SERVER']};"
            f"DATABASE={DB_CONFIG['DATABASE']};UID={DB_CONFIG['UID']};PWD={DB_CONFIG['PWD']}"
        ) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
        
            for idx, row in enumerate(cursor, start=1):
                print(f"Processing Record {idx}: ID = {row[0]}")
                try:
                    preprocessed_text = preprocess_text(row[1], row[2], row[3])
                    text_embedded = await w2c(preprocessed_text)
                    await store_qdrant(text_embedded, row[0])
                    print(f"Record {idx} processed successfully.")
                except Exception as e:
                    print(f"Error processing Record {idx}: {e}")
                finally:
                    gc.collect()  # Clear memory and garbage collect
                    
        return {"message": "Store to Qdrant successful"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during processing: {e}")

    

@app.post("/python/get-matching-projects")
async def infer_method1(input_data: TextInput):
    """
    Search for matching embeddings in Qdrant based on input text.
    """
    try:
        query = """
        SELECT 
            STRING_AGG(CONCAT(u.Education, ' ', u.Bio, ' ', u.Experience, ' ', s.SkillName, ' '), ' ') AS Profile
        FROM 
            AspNetUsers AS u
        JOIN 
            FreelancerSkill AS fs ON u.Id = fs.FreelancersId
        LEFT JOIN 
            Skill AS s ON fs.SkillsId = s.Id
        WHERE 
            u.Id = ?
        GROUP BY 
            u.Id, u.Education, u.Bio, u.Experience;
        """
        with pyodbc.connect(
                f"DRIVER={DB_CONFIG['DRIVER']};SERVER={DB_CONFIG['SERVER']};"
                f"DATABASE={DB_CONFIG['DATABASE']};UID={DB_CONFIG['UID']};PWD={DB_CONFIG['PWD']}"
        ) as conn:
            cursor = conn.cursor()
            cursor.execute(query, input_data.text)
            result = cursor.fetchall()
        
        if len(result) > 0:
            profiles = [row.Profile for row in result]
            # Join profiles into a single string
            combined_profiles = " ".join(profiles)
            text_embeds = await w2c(combined_profiles)
            matching_projects = await search_qdrant(text_embeds)
            return {"embeddings": matching_projects}
        else:
            return {"embeddings": ""}
       
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


async def w2c(input_data: str) -> np.ndarray:
    try:
        load_model()

        text_tensor = np.array([input_data], dtype=np.object_)

        # Prepare Triton input
        text_input = grpcclient.InferInput("TEXT", text_tensor.shape, "BYTES")
        # text_input = httplient.InferInput("TEXT", text_tensor.shape, "BYTES")
        text_input.set_data_from_numpy(text_tensor)

        # Prepare Triton output
        output = grpcclient.InferRequestedOutput("VEC")
        # output = httplient.InferRequestedOutput("VEC")

        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=[text_input],
            outputs=[output],
            model_version=MODEL_VERSION,
        )

        text_embeds = response.as_numpy("VEC")
        unload_model()
        return text_embeds[0]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


async def store_qdrant(text_embeds: np.ndarray, project_id: str):
    """
    Store embeddings in Qdrant.
    """
    qdrant_client.upsert(
        collection_name="jp4f-vector-db",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=text_embeds.tolist(),
                payload={"id": project_id},
            )
        ],
    )

async def search_qdrant(text_embeds: np.ndarray):
    """
    Search for matching embeddings in Qdrant.
    """
    search_result = qdrant_client.query_points(
        collection_name="jp4f-vector-db",
        query=text_embeds.tolist(),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=20,
        with_payload=True,
    )
    
    # Initialize lists for IDs and scores
    ids = []
    scores = []

    # Loop through search results and collect IDs and scores
    for point in search_result.points:
        ids.append(point.payload["id"])
        scores.append(point.score)

    # Create a query string with the collected IDs
    id_string = ", ".join(f"'{id}'" for id in ids)
    
    def create_query(id_string):
        query = f"""
        SELECT *
        FROM [JP4F].[dbo].[Project]
        WHERE Id IN ({id_string})
        """
        return query        

    query = create_query(id_string)
    
    with pyodbc.connect(
            f"DRIVER={DB_CONFIG['DRIVER']};SERVER={DB_CONFIG['SERVER']};"
            f"DATABASE={DB_CONFIG['DATABASE']};UID={DB_CONFIG['UID']};PWD={DB_CONFIG['PWD']}"
    ) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        
    # Combine the results with their corresponding scores
    combined_results = [(row, score) for row, score in zip(result, scores)]
    
    return combined_results




def preprocess_text(title: str, description: str, skills: str) -> str:
    """
    Merge and preprocess text fields for vectorization.
    """
    return f"{title} {description} {skills}".replace("-", ".").replace("\n", " ")

def load_model():
    """
    Load models into Triton server.
    """
    url = f"http://{LOAD_URL}/v2/repository/models/{MODEL_NAME}/load"
    response = requests.post(url, json={"model_name": MODEL_NAME})
    if response.status_code != 200:
        print(f"Error loading model {MODEL_NAME}: {response.text}")
    url = f"http://{LOAD_URL}/v2/repository/models/jina/load"
    response = requests.post(url, json={"model_name": 'jina'})
    if response.status_code != 200:
        print(f"Error loading model jina: {response.text}")

def unload_model():
    """
    Unload models from Triton server.
    """
    url = f"http://{LOAD_URL}/v2/repository/models/{MODEL_NAME}/unload"
    response = requests.post(url, json={"model_name": MODEL_NAME})
    if response.status_code != 200:
        print(f"Error loading model {MODEL_NAME}: {response.text}")
    url = f"http://{LOAD_URL}/v2/repository/models/jina/unload"
    response = requests.post(url, json={"model_name": 'jina'})
    if response.status_code != 200:
        print(f"Error loading model jina: {response.text}")





def main():
    parser = argparse.ArgumentParser(description="Run FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on.")
    args = parser.parse_args()

    # Run the FastAPI app
    print(f"Starting FastAPI server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

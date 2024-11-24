from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gensim.models.doc2vec import Doc2Vec

import re
from typing import List
import tritonclient.grpc as grpcclient
import numpy as np





# FastAPI app
app = FastAPI()

# Triton server configuration
TRITON_SERVER_URL = "localhost:1235"
MODEL_NAME = "doc2vec"

# Initialize the Triton client
try:
    triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Triton server: {e}")

# Load the local Doc2Vec model
try:
    MODEL_PATH = r"/Users/viet-quanggg/SelfProjects/JP4F/triton-inference-server/deploy-triton/doc2vec/1/model/model.model"  # Adjust this path based on your environment
    model = Doc2Vec.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load Doc2Vec model: {e}")





# Request model for validation
class InferenceRequest(BaseModel):
    texts: List[str]

# Preprocessing function
def preprocess_text(text: str) -> str:
    # Convert the text to lowercase
    text = text.lower()
    # Remove punctuation and non-alphabetic characters
    text = re.sub(r"[^a-z]", " ", text)
    # Remove numerical values
    text = re.sub(r"\d+", "", text)
    # Remove extra whitespaces
    text = " ".join(text.split())
    return text

@app.post("/infer-local")
async def infer_texts(request: InferenceRequest):
    try:
        # Preprocess input texts
        input_texts = [preprocess_text(text) for text in request.texts]

        # Perform inference for each text
        vectorized_outputs = [model.infer_vector(text.split()).tolist() for text in input_texts]

        return {"vectors": vectorized_outputs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

@app.post("/infer-triton")
async def infer_texts(request: InferenceRequest):
    try:
        # Extract batch of texts
        input_texts = request.texts

        # Prepare Triton input
        inputs = []
        inputs.append(grpcclient.InferInput("TEXT", [len(input_texts)], "BYTES"))
        text2np = np.array(input_texts, dtype=np.object_)
        inputs[0].set_data_from_numpy(text2np)

        # Prepare Triton output
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("VECTORIZE"))

        # Perform inference
        response = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

        # Retrieve output
        vectorized_outputs = response.as_numpy("VECTORIZE")
        return {"vectors": vectorized_outputs.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")



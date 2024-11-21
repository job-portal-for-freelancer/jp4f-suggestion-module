from gensim.models.doc2vec import Doc2Vec
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import re


# Triton Python backend utils
try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.


class TritonPythonModel:
    
    def initialize(self, args: Dict[str, str]) -> None:
        self.logger = pb_utils.Logger
        
        current_name: str = str(Path(args["model_repository"]).parent.absolute())
        self.model = Doc2Vec.load(current_name + "/doc2vec/1/model/model.model")


    def preprocess_text(self, text:str) -> str:
        # Convert the text to lowercase
        text = text.lower()

        # Remove punctuation from the text
        text = re.sub('[^a-z]', ' ', text)

        # Remove numerical values from the text
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespaces
        text = ' '.join(text.split())

        return text
    
    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":

        responses = []

        for request in requests:
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
                .as_numpy()
                .tolist()
            ]
            self.logger.log_info(f"Input infor: {query}")
            self.logger.log_info(f"Input data type: {type(query)}")


            input_text = self.preprocess_text(query[0])

            result_vec = self.model.infer_vector(input_text.split())

            tensor_output = pb_utils.Tensor('VECTORIZE', result_vec)
            inference_response = pb_utils.InferenceResponse(output_tensors=[tensor_output])
            responses.append(inference_response)

        return responses



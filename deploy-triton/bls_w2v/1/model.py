import numpy as np
from transformers import AutoTokenizer, PretrainedConfig
from pathlib import Path
from typing import Callable, List, Optional, Union, Dict

# import torch
# Triton Python backend utils
try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]):
        current_name: str = str(Path(args["model_repository"]).parent.absolute())

        self.tokenizer = AutoTokenizer.from_pretrained(current_name + "/bls_w2v/1/configs/")
        self.config = PretrainedConfig.from_pretrained(current_name + "/bls_w2v/1/configs/")

        self.task_type = 'text-matching'
        self.task_id = np.array(self.config.lora_adaptations.index(self.task_type), dtype=np.int64)

        self.logger = pb_utils.Logger


    def mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray):
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        responses = []
        
        for request in requests:
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
                .as_numpy()
                .tolist()
            ]
        
        input_text = self.tokenizer(query, return_tensors='np')
  
        inputs = {
            "input_ids": input_text["input_ids"].astype(np.int64),
            "attention_mask": input_text["attention_mask"].astype(np.int64),
            "task_id": np.expand_dims(self.task_id, axis=0)  # Add batch dimension
        }

        # Construct output tensors for Triton
        inputs_tensors = [
            pb_utils.Tensor("input_ids", inputs["input_ids"]),
            pb_utils.Tensor("attention_mask", inputs["attention_mask"]),
            pb_utils.Tensor("task_id", inputs["task_id"])
        ]

        self.logger.log_info(f"input_ids shape: {inputs['input_ids'].shape}")
        self.logger.log_info(f"attention_mask shape: {inputs['attention_mask'].shape}")
        self.logger.log_info(f"task_id shape: {inputs['task_id'].shape}")


        inference_request = pb_utils.InferenceRequest(
            model_name="jina", 
            requested_output_names=['text_embeds'],  
            inputs=inputs_tensors   ,
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message())
        else:
            text_embed_tensor = pb_utils.get_output_tensor_by_name(
                inference_response, "text_embeds"
            )
            # for CPU
            text_embeds_np = text_embed_tensor.as_numpy()
            
            # for GPU
            # text_embeds_np = text_embed_tensor.to_dlpack()
            # text_embeds_np = torch.utils.dlpack.from_dlpack(text_embeds_np).cpu().numpy()

        self.logger.log_info(f"text_embeds shape: {type(text_embeds_np)}")


        embeddings = self.mean_pooling(text_embeds_np, input_text["attention_mask"])
        embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        self.logger.log_info(f"Result: {embeddings}")
        
        tensor_output = pb_utils.Tensor('VEC', embeddings)
        inference_response = pb_utils.InferenceResponse(output_tensors=[tensor_output])
        responses.append(inference_response)

        return responses
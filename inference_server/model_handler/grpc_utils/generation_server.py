import os
from concurrent import futures

import torch

import grpc

from ...constants import GRPC_OPTIONS
from ...models import Model
from ...utils import create_generate_request, print_rank_n, ScoreRequest
from .proto import generation_pb2, generation_pb2_grpc


class GenerationServer(generation_pb2_grpc.GenerationServiceServicer):
    def __init__(self, model: Model) -> None:
        self.model = model

    def _unpack_proto_query_kwargs(self, query_kwargs):
        query_kwargs = {k: getattr(v, v.WhichOneof("oneof_values")) for k, v in query_kwargs.items()}
        return query_kwargs

    def Generate(self, request, context):
        text = [r for r in request.texts]
        generate_kwargs = self._unpack_proto_query_kwargs(request.generate_kwargs)

        request = create_generate_request(text=text, generate_kwargs=generate_kwargs)

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self.model.input_device = local_rank

        response = self.model.generate(request)

        if isinstance(response, Exception):
            # if exception occurs, we don't this subprocess to crash
            response = generation_pb2.GenerationResponse(error=str(response))
        else:
            response = generation_pb2.GenerationResponse(
                texts=response.text, 
                num_generated_tokens=response.num_generated_tokens,
                scores_b64=response.scores_b64,
                hidden_states_b64=response.hidden_states_b64,
            )

        return response
    
    def Score(self, request, context):
        text = [r for r in request.texts]
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self.model.input_device = local_rank
        
        request = ScoreRequest(text=text)
        response = self.model.score(request)
 
        if isinstance(response, Exception):
            # if exception occurs, we don't this subprocess to crash
            response = generation_pb2.ScoreResponse(error=str(response))
        else:
            response = generation_pb2.ScoreResponse(
                tokens = response.tokens
                scores = response.scores
            )

        return response       
        
        
        # tokenizer = self.model.tokenizer
        # tokenizer.pad_token = tokenizer.eos_token
        
        # # Tokenize the input text
        # inputs = tokenizer(request.text, return_tensors="pt", padding=True)
        # input_ids = inputs["input_ids"]
        
        # response = model(input_ids, labels = input_ids)
        

            


def serve(inference_pipeline, port):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=GRPC_OPTIONS,
    )
    generation_pb2_grpc.add_GenerationServiceServicer_to_server(GenerationServer(inference_pipeline), server)
    server.add_insecure_port(f"[::]:{port}")
    print_rank_n("About to start server")
    server.start()
    print_rank_n("Started")
    server.wait_for_termination()

import argparse
import os
from functools import partial
from typing import Union
import pickle as pkl
import base64

import numpy as np

import torch

import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import is_offline_mode

from ..utils import GenerateRequest, GenerateResponse, GenerationMixin, TokenizeRequest, TokenizeResponse, run_rank_n


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self.tokenizer = None
        self.pad = None
        self.model = None
        self.input_device = None
        self.max_input_length = args.max_input_length
        self.max_batch_size = args.max_batch_size

    def generate(self, request: GenerateRequest) -> Union[GenerateResponse, Exception]:
        try:
            check_batch_size(len(request.text), self.max_batch_size)

            input_tokens = self.tokenizer(
                request.text, 
                return_tensors="pt", 
                padding=True,
                return_token_type_ids=False)
            max_input_length_in_batch = input_tokens.input_ids[0].shape[0]

            check_max_input_length(max_input_length_in_batch, self.max_input_length)

            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(self.input_device)

            # TODO: at some point, we will want to get rid of the custom GenerateMixin class
            # and use the generate method from transformers directly. for that we need to be 
            # able to get the number of generated tokens from the output of the generate
            output = GenerationMixin(self.model).generate(
                **input_tokens,
                min_length=request.min_length,
                do_sample=request.do_sample,
                early_stopping=request.early_stopping,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                typical_p=request.typical_p,
                repetition_penalty=request.repetition_penalty,
                bos_token_id=request.bos_token_id,
                pad_token_id=request.pad_token_id,
                eos_token_id=request.eos_token_id,
                length_penalty=request.length_penalty,
                no_repeat_ngram_size=request.no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=request.encoder_no_repeat_ngram_size,
                num_return_sequences=request.num_return_sequences,
                max_time=request.max_time,
                max_new_tokens=request.max_new_tokens,
                decoder_start_token_id=request.decoder_start_token_id,
                diversity_penalty=request.diversity_penalty,
                forced_bos_token_id=request.forced_bos_token_id,
                forced_eos_token_id=request.forced_eos_token_id,
                exponential_decay_length_penalty=request.exponential_decay_length_penalty,
                output_scores=request.output_scores,
                output_hidden_states=request.output_hidden_states,
                return_dict_in_generate=True,
            )

            output_tokens = output.sequences
            num_generated_tokens = output.num_generated_tokens.tolist()
            scores_b64 = []
            hidden_states_b64 = []

            if request.remove_input_from_output:
                # the generate method's output includes input too. Remove input if
                # that is requested by the user
                output_tokens = [x[-i:] if i != 0 else [] for x, i in zip(output_tokens, num_generated_tokens)]

            if request.output_scores:
                # convert timestep-wise tensors to numpy array
                scores = [score.cpu().numpy() for score in output.scores]
                # stack timesteps and transpose batch dimension
                # TODO: we can reduce communication by prunning the score for padding tokens
                # currently we are sending all scores for all tokens
                scores = np.stack(scores, axis=1)
                # split first dimension into seperate tensors
                # binarize the tensors/arrays
                scores_b64 = [
                        base64.encodebytes(pkl.dumps(score)).decode("ascii") 
                        for score in scores
                ]

            if request.output_hidden_states:
                # take last timestep for each layer (since it includes all the hidden states
                # for the previous steps)
                layerwise_hidden_states = [
                        layer_hidden_states.cpu().numpy() 
                        for layer_hidden_states in output.hidden_states[-1]
                ]
                # reshape so as to have a tensor per batch element
                hidden_states = []
                for b in range(len(layerwise_hidden_states[0])):
                    hidden_states.append([
                            layer_hidden_states[b] for layer_hidden_states in layerwise_hidden_states
                    ])

                hidden_states_b64 = [
                        base64.encodebytes(pkl.dumps(hidden_state)).decode("ascii")
                        for hidden_state in hidden_states]

            output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

            res = GenerateResponse(
                text=output_text, 
                num_generated_tokens=num_generated_tokens,
                scores_b64=scores_b64,
                hidden_states_b64=hidden_states_b64,
            )
            return res
        except Exception as exception:
            return exception

    def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        response = self.tokenizer(request.text, padding=request.padding)
        return TokenizeResponse(token_ids=response.input_ids, attention_mask=response.attention_mask)


def get_downloaded_model_path(model_name: str):
    f = partial(
        snapshot_download,
        repo_id=model_name,
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
        # maybe move to safetensors in the future
        ignore_patterns=["*.safetensors", "*.msgpack", "*.h5", "*log*", "*evaluation*", "tensorboard"],
    )
    # download only on 1 process
    run_rank_n(f, barrier=True)
    # now since the snapshot is downloaded, pass the model_path to all processes
    return f()


def check_max_input_length(input_token_length: int, max_input_length: int) -> None:
    if max_input_length is None:
        return

    if input_token_length > max_input_length:
        raise Exception(f"max supported input length = {max_input_length} for now")


def check_batch_size(batch_size: int, max_batch_size: int) -> None:
    if max_batch_size is None:
        return

    if batch_size > max_batch_size:
        raise Exception(f"max supported batch size = {max_batch_size} for now")


# this is a hack for now
def get_hf_model_class(model_class: str) -> Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
    return getattr(transformers, model_class)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # TODO: llama relied on this hack, check if it is still needed
    if "llama" in model_name.lower():
        tokenizer.pad_token_id = 2
        
    tokenizer.padding_side = "left"

    return tokenizer

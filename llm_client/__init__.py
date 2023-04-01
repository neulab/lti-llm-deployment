from typing import Any, Union, List, Optional, Dict
from dataclasses import dataclass
import requests
import numpy as np
import numpy.typing as npt
import pickle as pkl
import base64


@dataclass
class Output:
    text: str
    scores: Optional[npt.NDArray[np.float16]]
    hidden_states: Optional[List[npt.NDArray[np.float16]]]

    def __str__(self) -> str:
        return self.text


class ServerError(Exception):
    pass


class Client:
    """A client for the LTI's LLM API."""

    def __init__(self, address: str = "tir-1-28", port: int = 5000) -> None:
        """Initialize the client.

        Args:
            address: The address of the server. Defaults to "tir-1-23", a node on the LTI's TIR cluster.
            port: The port of the server. Defaults to 5000.
        """
        self.address = address
        self.port = port
        self.url = f"http://{self.address}:{self.port}"

    def prompt(
        self,
        text: Union[str, List[str]],
        max_new_tokens: int = 64,
        output_scores: bool = False,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> List[Output]:
        """Prompt the LLM currently being served with a text and return the response.
        Args:
            text: The text to prompt the LLM with.
            do_sample: Whether to use greedy decoding. Defaults to False.
            max_new_tokens: The maximum number of tokens to generate. Defaults to 64.
            output_scores: Whether to return the scores for each token. Defaults to False.
            output_hidden_states: Whether to return the hidden states for each token. Defaults to False.
            **kwargs: Additional keyword arguments to pass to model.
                They follow HF's generate API
        Returns:
        """
        if isinstance(text, str):
            return self.prompt([text], max_new_tokens, **kwargs)

        # TODO: Check for max length limit to avoid OOMs

        request_body = {
            "text": text,
            "max_new_tokens": max_new_tokens,
            "output_scores": output_scores,
            "output_hidden_states": output_hidden_states,
            **kwargs,
        }
        response: Dict[str, Any] = requests.post(
            url=f"{self.url}/generate/", json=request_body, verify=False
        ).json()

        if "error" in response:
            raise ServerError(
                f"Server-side Error -- {response['error']}: {response['message']}"
            )

        outputs = [
            Output(text=text, scores=None, hidden_states=None)
            for text in response["text"]
        ]

        if output_scores:
            assert response["scores_b64"] is not None
            scores = [
                pkl.loads(base64.decodebytes(score_b64.encode("ascii")))
                for score_b64 in response["scores_b64"]
            ]
            for output, score in zip(outputs, scores):
                output.scores = score

        if output_hidden_states:
            assert response["hidden_states_b64"] is not None
            hidden_states = [
                pkl.loads(base64.decodebytes(hidden_state_b64.encode("ascii")))
                for hidden_state_b64 in response["hidden_states_b64"]
            ]
            for output, hidden_state in zip(outputs, hidden_states):
                output.hidden_states = hidden_state

        return outputs

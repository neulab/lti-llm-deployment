from typing import Any, Union, List, Optional
from dataclasses import dataclass
import requests
import numpy as np
import numpy.typing as npt


# TODO: Not used yet since we are just returning strings at the moment
@dataclass
class Response:
    text: str
    scores: Optional[npt.NDArray[np.float16]]
    hidden_states: Optional[npt.NDArray[np.float16]]


class Client:
    """A client for the LTI's LLM API."""

    def __init__(self, address: str = "tir-1-23", port: int = 5000) -> None:
        """Initialize the client.

        Args:
            address: The address of the server. Defaults to "tir-1-7", a node on the LTI's TIR cluster.
            port: The port of the server. Defaults to 5000.
        """
        self.address = address
        self.port = port
        self.url = f"http://{self.address}:{self.port}"

    def prompt(
        self,
        text: Union[str, List[str]],
        max_new_tokens: int = 64,
        **kwargs: Any,
    ) -> Any:
        """Prompt the LLM currently being served with a text and return the response.
        Args:
            text: The text to prompt the LLM with.
            max_tokens: The maximum number of tokens to generate. Note
                that this is *excluding* the prompt tokens. Defaults to 64.
            do_sample: Whether to use greedy decoding. Defaults to False.
            **kwargs: Additional keyword arguments to pass to model.
                They follow HF's generate API
        Returns:
        """
        if isinstance(text, str):
            return self.prompt([text], max_new_tokens, **kwargs)[0]

        # TODO: Check for max length limit to avoid OOMs

        request_body = {
            "text": text,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }
        response = requests.post(
            url=f"{self.url}/generate/", json=request_body, verify=False
        )
        return response.json()["text"]

from typing import Dict, Any, Union, Iterable, List
import requests


class VectorizeModel:
    """
    A concrete class for vectorizing models using OpenAI embedding services.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1/",
        vector_dimensions: int = None,
        timeout: float = None,
    ):
        """
        Initializes the VectorizeModel instance.

        Args:
            model (str, optional): The model to use for embedding. Defaults to "text-embedding-3-small".
            api_key (str, optional): The API key for accessing the OpenAI service. Defaults to "".
            base_url (str, optional): The base URL for the OpenAI service. Defaults to "https://api.openai.com/v1/".
            vector_dimensions (int, optional): The number of dimensions for the embedding vectors. Defaults to None.
            timeout (float, optional): The timeout for the API request. Defaults to None.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.vector_dimensions = vector_dimensions  # Note: This may not be necessary if the dimension is provided by the model
        self.timeout = timeout

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'VectorizeModel':
        """
        Creates an instance of VectorizeModel from a configuration.

        Args:
            config (Dict[str, Any]): The configuration dictionary, similar to a JSON object.

        Returns:
            VectorizeModel: An instance of VectorizeModel.
        """
        api_key = config.get('api_key', '')
        base_url = config.get('base_url', 'https://api.openai.com/v1/')
        model = config.get('model', 'text-embedding-3-small')
        vector_dimensions = config.get('vector_dimensions')  # May not be necessary
        timeout = config.get('timeout')
        return cls(model, api_key, base_url, vector_dimensions, timeout)

    def vectorize(
        self, texts: Union[str, Iterable[str]]
    ) -> Union[List[float], Iterable[List[float]]]:
        """
        Vectorizes a text string into an embedding vector or multiple text strings into multiple embedding vectors.

        Args:
            texts (Union[str, Iterable[str]]): The text or texts to vectorize.

        Returns:
            Union[List[float], Iterable[List[float]]]: The embedding vector(s) of the text(s).
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if isinstance(texts, str):
            texts = [texts]

        response = requests.post(
            f"{self.base_url}/embeddings",
            json={"input": texts, "model": self.model},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        results = response.json().get('data', [])
        vectors = [item.get('embedding', []) for item in results]

        if len(vectors) != len(texts):
            raise ValueError("Number of results does not match the number of input texts")

        if isinstance(texts, str):
            return vectors[0]
        else:
            return vectors
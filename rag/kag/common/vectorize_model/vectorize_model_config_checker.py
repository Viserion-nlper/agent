
import json


class VectorizeModelConfigChecker:
    """
    A class that checks whether the vectorizer configuration is valid.

    This class provides a method to validate the vectorizer configuration and return the embedding vector dimensions if valid.
    """

    def check(self, vectorizer_config: str) -> int:
        """
        Checks the vectorizer configuration.

        If the configuration is valid, it returns the actual embedding vector dimensions.
        If the configuration is invalid, it raises a RuntimeError exception.

        Args:
            vectorizer_config (str): The vectorizer configuration to be checked.

        Returns:
            int: The embedding vector dimensions.

        Raises:
            RuntimeError: If the configuration is invalid.
        """
        try:
            config = json.loads(vectorizer_config)
            from rag.kag.common.vectorize_model.VectorizeModel import VectorizeModel

            vectorizer = VectorizeModel.from_config(config)
            res = vectorizer.vectorize("hello")
            return len(res)
        except Exception as ex:
            message = "invalid vectorizer config: %s" % str(ex)
            raise RuntimeError(message) from ex

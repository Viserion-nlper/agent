import json


class LLMConfigChecker(object):
    """
    Check whether the llm config is valid.
    """

    def check(self, config: str) -> str:
        """
        Check the llm config.

        * If the config is valid, return the generated text.

        * If the config is invalid, raise a RuntimeError exception.

        :param config: llm config
        :type config: str
        :return: the generated text
        :rtype: str
        :raises RuntimeError: if the config is invalid
        """
        from rag.kag.common.llm.openai_client import OpenAIClient

        config = json.loads(config)
        llm_client = OpenAIClient.from_config(config)
        try:
            res = llm_client("who are you?")
            return res
        except Exception as ex:
            raise RuntimeError(f"invalid llm config: {config}, for details: {ex}")


if __name__ == "__main__":
    # 测试
    config = {
        "api_key": "sk-2ba1960eb13f42fe9da1cb158041fa2e",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "type": "maas"
    }
    from openai_client import OpenAIClient
    llm_client = OpenAIClient.from_config(config)
    response = llm_client("hi 你好呀")
    print(response)


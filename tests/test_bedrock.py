import json

from app.llm import BedrockProvider


class FakeBody:
    def __init__(self, data): self._data = data
    def read(self): return self._data


class FakeBedrockClient:
    def __init__(self):
        self.invoke_calls = []
        self.converse_calls = []

    def invoke_model(self, modelId, body):
        self.invoke_calls.append((modelId, body))
        return {"body": FakeBody(json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode())}

    def converse(self, modelId, messages):
        self.converse_calls.append((modelId, messages))
        return {"output": {"message": {"content": [{"text": "generated answer"}]}}}


def test_embed_shapes_request_and_parses_embedding():
    c = FakeBedrockClient()
    p = BedrockProvider("titan-embed", "claude-gen", "us-east-1", client=c)
    assert p.embed("hello") == [0.1, 0.2, 0.3]
    model_id, body = c.invoke_calls[0]
    assert model_id == "titan-embed"
    assert json.loads(body) == {"inputText": "hello"}


def test_generate_uses_converse_and_parses_text():
    c = FakeBedrockClient()
    p = BedrockProvider("titan-embed", "claude-gen", "us-east-1", client=c)
    assert p.generate("why is the sky blue?") == "generated answer"
    model_id, messages = c.converse_calls[0]
    assert model_id == "claude-gen"
    assert messages == [{"role": "user", "content": [{"text": "why is the sky blue?"}]}]


def test_model_name_is_gen_model():
    p = BedrockProvider("titan-embed", "claude-gen", "us-east-1", client=FakeBedrockClient())
    assert p.model_name == "claude-gen"

import json
import ssl
import time

import requests
import truststore
from langsmith import traceable
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ProviderError(RuntimeError):
    pass


class SystemTrustAdapter(HTTPAdapter):
    def __init__(self, **kwargs):
        self.context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        super().__init__(**kwargs)

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        host, options = super().build_connection_pool_key_attributes(request, verify, cert)
        if verify is True:
            options["ssl_context"] = self.context
        return host, options


class OpenRouter:
    def __init__(self, settings):
        self.settings = settings
        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Bearer {settings.key}"
        retry = Retry(total=1, status_forcelist=[429, 502, 503, 504],
                      allowed_methods={"POST"}, backoff_factor=0.5, respect_retry_after_header=False)
        self.http.mount("https://", SystemTrustAdapter(max_retries=retry))

    @traceable(name="OpenRouter request", run_type="llm",
               process_inputs=lambda inputs: {"endpoint": inputs.get("endpoint"),
                   "model": inputs.get("body", {}).get("model")})
    def request(self, endpoint, body, usage):
        start = time.perf_counter()
        record = {"endpoint": endpoint, "requested_model": body["model"], "ok": False}
        try:
            response = self.http.post(f"https://openrouter.ai/api/v1/{endpoint}", json=body,
                                      timeout=(5, 30))
            record["http_status"] = response.status_code
            if not response.ok:
                try:
                    error = response.json().get("error", {})
                    record["error"] = {"code": error.get("code"), "message": str(error.get("message", ""))[:300]}
                except (ValueError, AttributeError):
                    record["error"] = {"message": "Non-JSON provider error"}
            response.raise_for_status()
            result = response.json()
            if "error" in result:
                raise ProviderError("Model provider returned an error.")
            record.update(ok=True, model=result.get("model"), provider=result.get("provider"),
                          usage=result.get("usage") or {})
            return result
        except (requests.RequestException, ValueError) as exc:
            raise ProviderError("The model provider could not complete the request.") from exc
        finally:
            record["seconds"] = round(time.perf_counter() - start, 4)
            usage.append(record)

    def embed(self, texts, usage):
        raw = self.request("embeddings", {"model": self.settings.embedding_model, "input": texts}, usage)
        try:
            items = sorted(raw["data"], key=lambda row: row["index"])
            if [row["index"] for row in items] != list(range(len(texts))):
                raise ValueError("Unexpected embedding indexes")
            return [row["embedding"] for row in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("Model provider returned invalid embeddings.") from exc

    def json_chat(self, system, content, schema, usage):
        if self.settings.chat_model.startswith("openai/"):
            # OpenAI strict mode requires each field in these flat response schemas.
            schema = {**schema, "required": list(schema["properties"])}
        req = {
            "model": self.settings.chat_model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(content, ensure_ascii=False)}],
            "response_format": {"type": "json_schema", "json_schema":
                                {"name": "urbancart_output", "strict": True, "schema": schema}},
            "provider": {"require_parameters": True},
            "max_tokens": 2200,
            "temperature": 0,
        }
        try:
            result = self.request("chat/completions", req, usage)
            return json.loads(result["choices"][0]["message"]["content"])
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            raise ProviderError("Model provider returned invalid structured output.") from exc

    def tool_chat(self, messages, tools, usage):
        """Native model tool selection; execution remains inside our validated server."""
        result = self.request("chat/completions", {
            "model": self.settings.chat_model, "messages": messages, "tools": tools,
            "tool_choice": "auto",
            "temperature": 0, "max_tokens": 1400,
            "provider": {"require_parameters": True},
        }, usage)
        try:
            message = result["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("Invalid message")
            return {k: v for k, v in message.items() if k in {"role", "content", "tool_calls"}}
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("Invalid agent response.") from exc

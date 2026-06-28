import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from mempool.adapters import OpenAICompatibleClient, OpenAICompatibleConfig


class Handler(BaseHTTPRequestHandler):
    auth_header = ""
    post_body = {}

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self.send_response(404)
            self.end_headers()
            return
        Handler.auth_header = self.headers.get("Authorization", "")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"data": [{"id": "small"}, {"id": "strong"}]}).encode()
        )

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        Handler.post_body = json.loads(self.rfile.read(length).decode())
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        )

    def log_message(self, format: str, *args: object) -> None:
        return


class OpenAICompatibleAdapterTest(unittest.TestCase):
    def test_lists_models(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAICompatibleClient(
                OpenAICompatibleConfig(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1"
                )
            )
            self.assertEqual(client.list_models(), ("small", "strong"))
        finally:
            server.shutdown()
            server.server_close()

    def test_uses_api_key_env_for_authorization(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        old_value = os.environ.get("MEMPOOL_TEST_API_KEY")
        os.environ["MEMPOOL_TEST_API_KEY"] = "secret-test-key"
        try:
            client = OpenAICompatibleClient(
                OpenAICompatibleConfig(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    api_key_env="MEMPOOL_TEST_API_KEY",
                )
            )
            client.list_models()
            self.assertEqual(Handler.auth_header, "Bearer secret-test-key")
        finally:
            if old_value is None:
                os.environ.pop("MEMPOOL_TEST_API_KEY", None)
            else:
                os.environ["MEMPOOL_TEST_API_KEY"] = old_value
            server.shutdown()
            server.server_close()

    def test_merges_chat_options_into_request_body(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAICompatibleClient(
                OpenAICompatibleConfig(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    chat_options={"seed": 7, "top_p": 0.95},
                )
            )
            client.chat("small", [{"role": "user", "content": "hello"}])
            self.assertEqual(Handler.post_body["model"], "small")
            self.assertEqual(Handler.post_body["temperature"], 0)
            self.assertEqual(Handler.post_body["seed"], 7)
            self.assertEqual(Handler.post_body["top_p"], 0.95)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

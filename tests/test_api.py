"""Run: python3 -m unittest discover -s tests (requires httpx)."""

import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api import create_app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.loads = 0

        def answer(question, history):
            self.calls.append((question, history.copy()))
            if question == "fail":
                raise RuntimeError("Test generation failure")
            return "Example answer", [{"title": "Example source"}]

        def load():
            self.loads += 1
            return SimpleNamespace(
                answer_question=answer,
                MAX_HISTORY_TURNS=4,
                model=SimpleNamespace(device="cpu"),
            )

        self.app = create_app(load)
        self.context = TestClient(self.app, raise_server_exceptions=False)
        self.client = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def test_conversation_and_independent_clients(self):
        response = self.client.post("/chat", json={"question": "Computer science?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sources"][0]["title"], "Example source")
        self.client.post("/chat", json={"question": "Who is its head?", "history": data["history"]})
        self.assertEqual(self.calls[-1][1], data["history"])
        self.client.post("/chat", json={"question": "New conversation"})
        self.assertEqual(self.calls[-1][1], [])
        self.assertEqual(self.loads, 1)

    def test_chat_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("University assistant", response.text)
        self.assertIn("fetch('/chat'", response.text)

    def test_history_limit(self):
        history = []
        for i in range(6):
            response = self.client.post("/chat", json={"question": str(i), "history": history})
            self.assertEqual(response.status_code, 200)
            history = response.json()["history"]
        self.assertEqual(len(history), 8)
        self.assertEqual(history[0]["content"], "2")

    def test_validation(self):
        for payload in [
            {"question": "   "},
            {"question": "x", "history": [{"role": "system", "content": "x"}]},
            {"question": "x", "history": [{"role": "user", "content": "x"}]},
            {"question": "x", "history": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "x"}]},
        ]:
            self.assertEqual(self.client.post("/chat", json=payload).status_code, 422)
        self.assertEqual(self.calls, [])

    def test_busy_and_health(self):
        lock = self.app.state.generation_lock
        lock.acquire()
        try:
            self.assertEqual(self.client.post("/chat", json={"question": "x"}).status_code, 503)
            self.assertEqual(self.client.get("/health").json(), {"status": "ready", "device": "cpu"})
        finally:
            lock.release()

    def test_generation_failure_releases_lock(self):
        self.assertEqual(self.client.post("/chat", json={"question": "fail"}).status_code, 500)
        self.assertEqual(self.client.post("/chat", json={"question": "Retry"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()

"""
Automated Integration Tests for Flask REST API Endpoints.
Covers /api/chat, /api/faqs, /api/health, error handling, validation,
and status codes.
"""

import json
import unittest

from app import app


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for Flask API routes and validations."""

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("FAQ", data["service"])
        self.assertGreater(data["faqs_loaded"], 0)

    def test_get_faqs_endpoint(self):
        response = self.client.get("/api/faqs")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("categories", data)
        self.assertIn("suggested_questions", data)
        self.assertIn("faqs", data)
        self.assertGreater(len(data["categories"]), 5)
        self.assertGreater(len(data["suggested_questions"]), 0)

    def test_get_faqs_with_category_filter(self):
        response = self.client.get("/api/faqs?category=Admissions")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        for faq in data["faqs"]:
            self.assertEqual(faq["category"], "Admissions")

    def test_chat_exact_match(self):
        payload = {"message": "How can I apply for admission?"}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["matched"])
        self.assertEqual(data["category"], "Admissions")
        self.assertIn("admissions portal", data["answer"])
        self.assertGreaterEqual(data["confidence"], 0.9)

    def test_chat_paraphrase_match(self):
        payload = {"message": "what is the tuition fees structure?"}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["matched"])
        self.assertEqual(data["category"], "Tuition Fees")

    def test_chat_unrelated_query_triggers_fallback(self):
        payload = {"message": "how to bake sourdough bread?"}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data["matched"])
        self.assertIn("couldn't find a reliable answer", data["answer"].lower())

    def test_chat_empty_message_validation(self):
        payload = {"message": "    "}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_chat_missing_message_field(self):
        payload = {"query": "hello"}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Missing required field 'message'", data["error"])

    def test_chat_non_string_message(self):
        payload = {"message": 12345}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("must be a string", data["error"])

    def test_chat_non_json_content_type(self):
        response = self.client.post(
            "/api/chat",
            data="message=hello",
            content_type="application/x-www-form-urlencoded"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Expected application/json", data["error"])

    def test_chat_excessive_length_query(self):
        payload = {"message": "a" * 1001}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("exceeds maximum allowed length", data["error"])

    def test_verify_key_endpoint_empty(self):
        response = self.client.post(
            "/api/verify-key",
            data=json.dumps({"api_key": ""}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["valid"])

    def test_verify_key_endpoint_invalid(self):
        response = self.client.post(
            "/api/verify-key",
            data=json.dumps({"api_key": "invalid_fake_key_12345"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["valid"])

    def test_chat_gemini_mode(self):
        payload = {"message": "Explain quantum entanglement in 1 sentence", "mode": "gemini"}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["matched"])
        self.assertEqual(data["source"], "Google Gemini AI")

    def test_chat_unrelated_query_triggers_fallback(self):
        payload = {"message": "how to bake sourdough bread?", "mode": "faq"}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data["matched"])
        self.assertIn("couldn't find a reliable answer", data["answer"].lower())

    def test_chat_custom_threshold_override(self):
        payload_normal = {
            "message": "can you tell me about the campus shuttle?",
            "threshold": 0.20
        }
        res1 = self.client.post(
            "/api/chat",
            data=json.dumps(payload_normal),
            content_type="application/json"
        )
        self.assertEqual(res1.status_code, 200)
        self.assertTrue(res1.get_json()["matched"])

        payload_strict = {
            "message": "can you tell me about the campus shuttle?",
            "threshold": 0.95,
            "mode": "faq"
        }
        res2 = self.client.post(
            "/api/chat",
            data=json.dumps(payload_strict),
            content_type="application/json"
        )
        self.assertEqual(res2.status_code, 200)
        self.assertFalse(res2.get_json()["matched"])

    def test_web_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Universal AI & FAQ Assistant", html)
        self.assertIn("chat-messages", html)

    def test_404_api_route(self):
        response = self.client.get("/api/nonexistent")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn("Endpoint not found", data["error"])


if __name__ == "__main__":
    unittest.main()


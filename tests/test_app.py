"""
Test suite for T5 Dialogue Summarizer API
Tests preprocessing, schemas, health endpoint, samples endpoint, and validation rules.
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Add APP directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

from app import app, clean_data


class TestSummarizerApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_clean_data_logic(self):
        """Verify data normalization strips CRLF, HTML tags, and converts to lowercase."""
        raw_input = "Hello<br>World!\r\nThis is   a <b>Test</b>."
        cleaned = clean_data(raw_input)
        self.assertNotIn("<", cleaned)
        self.assertNotIn(">", cleaned)
        self.assertNotIn("\r\n", cleaned)
        self.assertEqual(cleaned, "hello world! this is a test.")

    def test_health_endpoint(self):
        """Verify health endpoint responds with 200 and expected schema."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("ready", data)
        self.assertIn("device", data)
        self.assertIn("torch_version", data)

    def test_samples_endpoint(self):
        """Verify sample dialogues endpoint returns valid list of samples."""
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        samples = response.json()
        self.assertIsInstance(samples, list)
        self.assertGreaterEqual(len(samples), 3)
        for s in samples:
            self.assertIn("title", s)
            self.assertIn("dialogue", s)
            self.assertGreater(len(s["dialogue"]), 10)

    def test_summarize_validation_empty_input(self):
        """Verify 422 Unprocessable Entity is raised when input dialogue is too short or empty."""
        response = self.client.post("/summarize/", json={"dialogue": ""})
        self.assertEqual(response.status_code, 422)

    def test_html_ui_renders(self):
        """Verify root endpoint serves HTML content."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Abstractive", response.text)


if __name__ == "__main__":
    unittest.main()


import unittest
import os
import sys

# Add project root and flask-app directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLASK_APP_DIR = os.path.join(PROJECT_ROOT, "flask-app")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if FLASK_APP_DIR not in sys.path:
    sys.path.insert(0, FLASK_APP_DIR)

# Import app from flask-app/app.py
from app import app


class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Configure app for testing mode
        app.testing = True
        cls.client = app.test_client()

    def test_home_page(self):
        """Test GET request to home page '/' renders correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200, "Home page should return status code 200")
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data, "Page title should match")
        self.assertIn(b'name="text"', response.data, "Text area input should be present")

    def test_predict_page_positive_review(self):
        """Test POST request to '/predict' with a strongly positive review."""
        positive_review = (
            "Brilliant direction and stunning visual effects! Even at nearly three hours, "
            "the pacing never falters. I was glued to my seat the entire time"
        )
        response = self.client.post('/predict', data={"text": positive_review})
        
        self.assertEqual(response.status_code, 200, "Predict endpoint should return status code 200")
        
        # In index.html, positive sentiment renders 'Positive Sentiment' and class 'positive'
        self.assertIn(
            b'Positive',
            response.data,
            f"Expected positive sentiment in response, but received:\n{response.data.decode('utf-8')}"
        )
        self.assertNotIn(
            b'Negative',
            response.data,
            "Response should not contain 'Negative' for a positive review"
        )

    def test_metrics_endpoint(self):
        """Test Prometheus '/metrics' endpoint exposes application metrics."""
        response = self.client.get('/metrics')
        self.assertEqual(response.status_code, 200, "Metrics endpoint should return status code 200")
        self.assertIn(b'app_request_count', response.data, "Metrics should contain app_request_count")


if __name__ == '__main__':
    unittest.main()
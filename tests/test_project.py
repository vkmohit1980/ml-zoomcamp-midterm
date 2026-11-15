import json
import unittest

from fastapi.testclient import TestClient

from data_source import ROOT
from predict import app
from train import LABELS, TARGET, load_data, split


class ProjectTests(unittest.TestCase):
    def test_dataset_and_chronology(self):
        data = load_data()
        train, valid, test = split(data)
        self.assertEqual(len(data), 10_129)
        self.assertEqual(set(train[TARGET]), set(LABELS))
        self.assertEqual(sum(map(len, (train, valid, test))), len(data))
        self.assertLess(train.timestamp.max(), valid.timestamp.min())
        self.assertLess(valid.timestamp.max(), test.timestamp.min())

    def test_review_request_and_validation(self):
        payload = json.loads((ROOT / "examples/request.json").read_text())
        with TestClient(app) as client:
            health = client.get("/health").json()
            self.assertEqual(set(health["features"]), set(payload))
            result = client.post("/predict", json=payload)
            self.assertEqual(result.status_code, 200)
            self.assertIn(result.json()["occupancy_count"], LABELS)
            self.assertEqual(result.json()["model"], health["model"])
            self.assertEqual(client.post("/predict", json={}).status_code, 422)
            self.assertEqual(client.post("/predict", json={**payload, "extra": 0}).status_code, 422)
            name = next(iter(payload))
            self.assertEqual(client.post("/predict", json={**payload, name: -1}).status_code, 422)

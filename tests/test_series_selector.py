import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from lib import series_selector as selector


class SeriesSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_df = pd.DataFrame({"date": ["2020-01-01"], "value": [1.0]})

    def test_candidate_queue_order_and_dedup(self) -> None:
        preferences = {"money_scale": ["MYAGM2JPM189S", {"id": "ALT", "start": "1980-01-01"}]}
        with patch.dict("os.environ", {"MONEY_SERIES": "ALT@1970-01-01"}, clear=True):
            queue = selector.candidate_queue(
                "money_scale",
                "MONEY_SERIES",
                preferences,
                {"money_scale": [{"id": "MYAGM2JPM189S", "title": "Default"}]},
            )
        self.assertEqual(queue[0]["id"], "ALT")
        self.assertEqual(queue[0]["start"], "1970-01-01")
        self.assertEqual(queue[0]["source"], "env:MONEY_SERIES")
        self.assertEqual(queue[1]["id"], "MYAGM2JPM189S")
        self.assertEqual(queue[1]["source"], "config")
        self.assertEqual(len(queue), 2)

    def test_select_series_uses_config_before_default(self) -> None:
        preferences = {"money_scale": [{"id": "BAD"}, {"id": "GOOD", "start": "1985-01-01"}]}

        def fake_fetch(series_id: str, start: str) -> pd.DataFrame:
            if series_id == "GOOD":
                return self.sample_df
            raise RuntimeError("fail")

        with patch.dict("os.environ", {}, clear=True):
            result = selector.select_series(
                "money_scale",
                "MONEY_SERIES",
                fake_fetch,
                preferences=preferences,
                defaults={"money_scale": [{"id": "FALLBACK"}]},
            )
        self.assertEqual(result["id"], "GOOD")
        self.assertEqual(result["start"], "1985-01-01")
        self.assertEqual(result["source"], "config")

    def test_select_series_falls_back_to_default(self) -> None:
        def fake_fetch(series_id: str, start: str) -> pd.DataFrame:
            if series_id == "DEFAULT":
                return self.sample_df
            raise RuntimeError("fail")

        with patch.dict("os.environ", {}, clear=True):
            result = selector.select_series(
                "money_scale",
                "MONEY_SERIES",
                fake_fetch,
                preferences={},
                defaults={"money_scale": [{"id": "DEFAULT", "start": "2000-01-01"}]},
            )
        self.assertEqual(result["id"], "DEFAULT")
        self.assertEqual(result["source"], "default")
        self.assertEqual(result["start"], "2000-01-01")

    def test_select_series_no_candidate_raises(self) -> None:
        def fake_fetch(series_id: str, start: str) -> pd.DataFrame:
            raise RuntimeError("fail")

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                selector.select_series(
                    "money_scale",
                    "MONEY_SERIES",
                    fake_fetch,
                    preferences={},
                    defaults={"money_scale": []},
                )

    def test_load_series_preferences_parses_yaml(self) -> None:
        yaml_content = (
            "series:\n"
            "  money_scale:\n"
            "    preferred:\n"
            "      - id: ALT\n"
            "        start: 1980-01-01\n"
            "  base_proxy:\n"
            "    - BASE1\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(yaml_content)
            tmp_path = tmp.name
        try:
            prefs = selector.load_series_preferences(tmp_path)
        finally:
            os.unlink(tmp_path)
        self.assertIn("money_scale", prefs)
        self.assertIsInstance(prefs["money_scale"], list)
        self.assertEqual(prefs["money_scale"][0]["id"], "ALT")
        self.assertIn("base_proxy", prefs)
        self.assertEqual(prefs["base_proxy"], ["BASE1"])


if __name__ == "__main__":
    unittest.main()

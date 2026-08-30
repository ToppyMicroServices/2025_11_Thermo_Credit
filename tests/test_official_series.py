from __future__ import annotations

import json

from lib.official_series import fetch_ecb_series, fetch_fred_series


class _Response:
    def __init__(self, *, text: str = "", payload=None):
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_fetch_fred_series_uses_public_csv_without_key():
    session = _Session(_Response(text="observation_date,TEST\n2025-01-01,1.5\n"))
    frame = fetch_fred_series("TEST", start="2025-01-01", session=session)

    assert frame["value"].tolist() == [1.5]
    assert session.calls[0][1]["params"] == {"id": "TEST", "cosd": "2025-01-01"}


def test_fetch_fred_series_uses_json_api_with_key():
    payload = {"observations": [{"date": "2025-01-01", "value": "2.5"}]}
    session = _Session(_Response(payload=payload))
    frame = fetch_fred_series("TEST", api_key="secret", session=session)

    assert frame["value"].tolist() == [2.5]
    assert session.calls[0][1]["params"]["api_key"] == "secret"


def test_fetch_ecb_series_normalizes_csvdata():
    text = "TIME_PERIOD,OBS_VALUE,TITLE\n2025-01,10.0,Example\n2025-02,11.0,Example\n"
    session = _Session(_Response(text=text))
    frame = fetch_ecb_series("BSI", "key", start="2025-01", session=session)

    assert frame["value"].tolist() == [10.0, 11.0]
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-01", "2025-02-01"]
    assert session.calls[0][1]["params"]["format"] == "csvdata"

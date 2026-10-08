import pytest

from recompete import download


def test_failed_usaspending_job_is_requeued(monkeypatch):
    calls = {"n": 0}

    def request_download(start, end, date_type="date_signed"):
        calls["n"] += 1
        return {"status_url": f"https://example.test/{calls['n']}"}

    def wait_for(url, label):
        if url.endswith("/1"):
            raise RuntimeError(f"{label} download failed: An error occurred.")
        return {"status": "finished", "file_url": "https://example.test/file"}

    monkeypatch.setattr(download, "request_download", request_download)
    monkeypatch.setattr(download, "wait_for", wait_for)
    monkeypatch.setattr(download.time, "sleep", lambda _seconds: None)

    status = download.fetch_status(
        "2026-09-21", "2026-10-05", "last_modified_date", "modified 2026-09-21..2026-10-05"
    )

    assert status["status"] == "finished"
    assert calls["n"] == 2


def test_failed_usaspending_job_raises_after_three_attempts(monkeypatch):
    calls = {"n": 0}

    def request_download(start, end, date_type="date_signed"):
        calls["n"] += 1
        return {"status_url": f"https://example.test/{calls['n']}"}

    def wait_for(url, label):
        raise RuntimeError(f"{label} download failed: An error occurred.")

    monkeypatch.setattr(download, "request_download", request_download)
    monkeypatch.setattr(download, "wait_for", wait_for)
    monkeypatch.setattr(download.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="An error occurred"):
        download.fetch_status("2026-09-21", "2026-10-05", "last_modified_date", "modified")

    assert calls["n"] == 3

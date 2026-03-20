from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional

import requests


def _fail(message: str) -> NoReturn:
    raise SystemExit(message)


def _clean_api_url(raw: str) -> str:
    return raw.rstrip("/")


def _extract_deposition_id(url: str) -> int:
    match = re.search(r"/deposit/depositions/(\d+)$", url)
    if not match:
        _fail(f"Could not parse deposition id from latest_draft URL: {url}")
    return int(match.group(1))


def _extract_numeric_id(raw: str) -> str:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return ""
    if cleaned.isdigit():
        return cleaned
    match = re.search(r"zenodo\.(\d+)$", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"/records/(\d+)$", cleaned)
    if match:
        return match.group(1)
    _fail(
        "Could not extract a Zenodo record id from "
        f"{cleaned}. Use a numeric id, DOI, or /records/<id> URL."
    )


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    expected: Optional[List[int]] = None,
    **kwargs: Any,
) -> Any:
    response = session.request(method, url, timeout=60, **kwargs)
    if expected is not None and response.status_code not in expected:
        body = response.text[:4000]
        _fail(f"{method} {url} failed with {response.status_code}: {body}")
    if response.status_code == 204:
        return None
    return response.json()


def _latest_published_deposition(
    session: requests.Session,
    api_url: str,
    concept_record_id: str,
) -> Dict[str, Any]:
    results = _request_json(
        session,
        "GET",
        f"{api_url}/deposit/depositions",
        expected=[200],
        params={
            "q": f"conceptrecid:{concept_record_id}",
            "status": "published",
            "sort": "mostrecent",
            "all_versions": 1,
            "size": 1,
        },
    )
    if not results:
        _fail(
            "No published Zenodo deposition found for concept record id "
            f"{concept_record_id}. Check ZENODO_CONCEPT_RECORD_ID."
        )
    return results[0]


def _resolve_concept_record_id(
    session: requests.Session,
    api_url: str,
    *,
    explicit_concept_record_id: str,
    seed_record_id_or_doi: str,
) -> str:
    if explicit_concept_record_id:
        return explicit_concept_record_id
    if not seed_record_id_or_doi:
        _fail("ZENODO_CONCEPT_RECORD_ID or ZENODO_SEED_RECORD_ID_OR_DOI is required.")

    record_id = _extract_numeric_id(seed_record_id_or_doi)
    deposition = _request_json(
        session,
        "GET",
        f"{api_url}/deposit/depositions/{record_id}",
        expected=[200],
    )
    concept_record_id = str(deposition.get("conceptrecid") or "")
    if not concept_record_id:
        _fail(
            "Zenodo record "
            f"{record_id} did not expose conceptrecid. Use ZENODO_CONCEPT_RECORD_ID explicitly."
        )
    return concept_record_id


def _existing_draft_deposition(
    session: requests.Session,
    api_url: str,
    concept_record_id: str,
) -> Optional[Dict[str, Any]]:
    results = _request_json(
        session,
        "GET",
        f"{api_url}/deposit/depositions",
        expected=[200],
        params={
            "q": f"conceptrecid:{concept_record_id}",
            "status": "draft",
            "sort": "mostrecent",
            "all_versions": 1,
            "size": 1,
        },
    )
    return results[0] if results else None


def _ensure_draft(
    session: requests.Session,
    api_url: str,
    concept_record_id: str,
) -> Dict[str, Any]:
    draft = _existing_draft_deposition(session, api_url, concept_record_id)
    if draft is not None:
        return draft

    latest = _latest_published_deposition(session, api_url, concept_record_id)
    created = _request_json(
        session,
        "POST",
        f"{api_url}/deposit/depositions/{latest['id']}/actions/newversion",
        expected=[201],
    )
    latest_draft = str(created.get("links", {}).get("latest_draft") or "")
    draft_id = _extract_deposition_id(latest_draft)
    return _request_json(
        session,
        "GET",
        f"{api_url}/deposit/depositions/{draft_id}",
        expected=[200],
    )


def _maybe_update_description(description: str, tag: str, release_url: str) -> str:
    if not release_url:
        return description
    marker = f"GitHub release: <a href=\"{release_url}\">{tag}</a>"
    if marker in description:
        return description
    suffix = f'<p>GitHub release: <a href="{release_url}">{tag}</a></p>'
    return description + suffix if description else suffix


def _update_metadata(
    session: requests.Session,
    api_url: str,
    draft: Dict[str, Any],
    *,
    tag: str,
    release_url: str,
) -> Dict[str, Any]:
    metadata = dict(draft.get("metadata") or {})
    version = tag.split("/")[-1]
    metadata["version"] = version
    description = str(metadata.get("description") or "")
    metadata["description"] = _maybe_update_description(description, tag, release_url)
    return _request_json(
        session,
        "PUT",
        f"{api_url}/deposit/depositions/{draft['id']}",
        expected=[200],
        json={"metadata": metadata},
        headers={"Content-Type": "application/json"},
    )


def _delete_matching_files(
    session: requests.Session,
    api_url: str,
    draft: Dict[str, Any],
    *,
    filename: str,
) -> None:
    for file_info in draft.get("files") or []:
        file_id = file_info.get("id")
        key = str(file_info.get("filename") or file_info.get("key") or file_info.get("name") or "")
        if file_id and key == filename:
            _request_json(
                session,
                "DELETE",
                f"{api_url}/deposit/depositions/{draft['id']}/files/{file_id}",
                expected=[204],
            )


def _upload_file(
    session: requests.Session,
    draft: Dict[str, Any],
    *,
    file_path: Path,
) -> Dict[str, Any]:
    bucket_url = str(draft.get("links", {}).get("bucket") or "")
    if not bucket_url:
        _fail("Zenodo draft is missing a bucket link.")
    with file_path.open("rb") as handle:
        response = session.put(
            f"{bucket_url}/{file_path.name}",
            data=handle,
            timeout=300,
        )
    if response.status_code not in (200, 201):
        _fail(
            "Uploading theory.pdf to Zenodo failed with "
            f"{response.status_code}: {response.text[:4000]}"
        )
    return response.json()


def _publish(
    session: requests.Session,
    api_url: str,
    draft_id: int,
) -> Dict[str, Any]:
    return _request_json(
        session,
        "POST",
        f"{api_url}/deposit/depositions/{draft_id}/actions/publish",
        expected=[202],
    )


def _write_summary(published: Dict[str, Any]) -> None:
    lines = ["## Zenodo update", ""]
    doi = str(published.get("doi") or published.get("metadata", {}).get("prereserve_doi", {}).get("doi") or "")
    html_url = str(published.get("links", {}).get("html") or "")
    record_id = str(published.get("record_id") or published.get("id") or "")
    if record_id:
        lines.append(f"- Record id: `{record_id}`")
    if doi:
        lines.append(f"- DOI: `{doi}`")
    if html_url:
        lines.append(f"- URL: {html_url}")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the release theory.pdf to Zenodo and publish a new version.")
    parser.add_argument("--pdf", default="tex/theory.pdf", help="Path to the theory PDF to upload.")
    parser.add_argument("--tag", required=True, help="Git tag or release tag name.")
    parser.add_argument("--release-url", default="", help="GitHub release URL to append to the Zenodo description.")
    parser.add_argument(
        "--api-url",
        default=os.getenv("ZENODO_API_URL", "https://zenodo.org/api"),
        help="Zenodo API base URL.",
    )
    parser.add_argument(
        "--concept-record-id",
        default=os.getenv("ZENODO_CONCEPT_RECORD_ID", ""),
        help="Zenodo concept record id that should receive new versions.",
    )
    parser.add_argument(
        "--seed-record-id-or-doi",
        default=os.getenv("ZENODO_SEED_RECORD_ID_OR_DOI", ""),
        help="A current Zenodo record id, DOI, or /records/<id> URL used to resolve conceptrecid automatically.",
    )
    parser.add_argument(
        "--access-token",
        default=os.getenv("ZENODO_ACCESS_TOKEN", ""),
        help="Zenodo personal access token.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        _fail(f"theory PDF not found: {pdf_path}")
    if not args.access_token:
        _fail("ZENODO_ACCESS_TOKEN is required.")
    api_url = _clean_api_url(args.api_url or "https://zenodo.org/api")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {args.access_token}"})
    concept_record_id = _resolve_concept_record_id(
        session,
        api_url,
        explicit_concept_record_id=args.concept_record_id,
        seed_record_id_or_doi=args.seed_record_id_or_doi,
    )

    draft = _ensure_draft(session, api_url, concept_record_id)
    draft = _update_metadata(
        session,
        api_url,
        draft,
        tag=args.tag,
        release_url=args.release_url,
    )
    _delete_matching_files(session, api_url, draft, filename=pdf_path.name)
    _upload_file(session, draft, file_path=pdf_path)
    published = _publish(session, api_url, int(draft["id"]))
    _write_summary(published)


if __name__ == "__main__":
    main()

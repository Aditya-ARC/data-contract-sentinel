"""A small GitHub REST client for incidents (standard library only).

Only the calls the agent needs: find an open incident, create an Issue, create
a pull request, and label things. The HTTP transport is injectable for tests.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, bytes]]


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"GitHub API error {status}: {message}")
        self.status = status
        self.message = message


def _urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None):
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class GitHubClient:
    def __init__(
        self,
        token: str,
        repo: str,
        api_url: str = "https://api.github.com",
        transport: Transport | None = None,
    ) -> None:
        if "/" not in repo:
            raise ValueError("repo must look like owner/name")
        self.repo = repo
        self._api = api_url.rstrip("/")
        self._token = token
        self._transport = transport or _urllib_transport

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "data-contract-sentinel",
        }
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        status, raw = self._transport(method, f"{self._api}{path}", headers, payload)
        data = json.loads(raw) if raw else {}
        if status >= 400:
            message = data.get("message", "") if isinstance(data, dict) else str(data)
            raise GitHubError(status, message)
        return data

    def default_branch(self) -> str:
        return self._request("GET", f"/repos/{self.repo}")["default_branch"]

    def find_open_incident(self, marker: str, label: str) -> dict[str, Any] | None:
        """Find an open Issue or PR carrying this label and hidden marker."""
        items = self._request(
            "GET", f"/repos/{self.repo}/issues?state=open&labels={label}&per_page=100"
        )
        for item in items:
            if marker in (item.get("body") or ""):
                return item
        return None

    def ensure_label(self, name: str, color: str, description: str) -> None:
        """Best effort: labels are cosmetic, so failures are ignored."""
        try:
            self._request(
                "POST",
                f"/repos/{self.repo}/labels",
                {"name": name, "color": color, "description": description},
            )
        except GitHubError:
            pass

    def create_issue(self, title: str, body: str, labels: list[str]) -> dict[str, Any]:
        return self._request(
            "POST", f"/repos/{self.repo}/issues", {"title": title, "body": body, "labels": labels}
        )

    def create_pull_request(
        self, title: str, body: str, head: str, base: str, labels: list[str]
    ) -> dict[str, Any]:
        pr = self._request(
            "POST",
            f"/repos/{self.repo}/pulls",
            {"title": title, "body": body, "head": head, "base": base},
        )
        if labels:
            try:
                self._request("POST", f"/repos/{self.repo}/issues/{pr['number']}/labels", {"labels": labels})
            except GitHubError:
                pass  # the PR exists; a missing label should not undo it
        return pr

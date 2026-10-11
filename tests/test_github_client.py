from __future__ import annotations

import json

import pytest

from contractsentinel.agent.github import GitHubClient, GitHubError


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, json.loads(body) if body else None))
        status, data = self.responses.pop(0)
        return status, json.dumps(data).encode()


def client(*responses):
    rec = Recorder(*responses)
    return GitHubClient("tok", "owner/repo", transport=rec), rec


def test_repo_must_be_owner_slash_name():
    with pytest.raises(ValueError):
        GitHubClient("t", "justname")


def test_requests_carry_auth_headers():
    gh, rec = client((200, {"default_branch": "trunk"}))
    assert gh.default_branch() == "trunk"
    method, url, headers, _ = rec.calls[0]
    assert (method, url) == ("GET", "https://api.github.com/repos/owner/repo")
    assert headers["Authorization"] == "Bearer tok" and "github+json" in headers["Accept"]


def test_find_open_incident_matches_marker_in_body():
    items = [{"body": "other"}, {"body": "text <!-- m1 --> more", "html_url": "u"}, {"body": None}]
    gh, rec = client((200, items), (200, items))
    assert gh.find_open_incident("<!-- m1 -->", "data-contract-incident")["html_url"] == "u"
    assert "labels=data-contract-incident" in rec.calls[0][1]
    assert gh.find_open_incident("<!-- nope -->", "data-contract-incident") is None


def test_create_issue_posts_title_body_labels():
    gh, rec = client((201, {"number": 3}))
    assert gh.create_issue("T", "B", ["l"])["number"] == 3
    assert rec.calls[0][3] == {"title": "T", "body": "B", "labels": ["l"]}


def test_create_pull_request_then_labels():
    gh, rec = client((201, {"number": 9}), (200, []))
    gh.create_pull_request("T", "B", "head", "main", ["a", "b"])
    assert rec.calls[0][3] == {"title": "T", "body": "B", "head": "head", "base": "main"}
    assert rec.calls[1][1].endswith("/issues/9/labels") and rec.calls[1][3] == {"labels": ["a", "b"]}


def test_label_failure_does_not_undo_the_pr():
    gh, _ = client((201, {"number": 9}), (404, {"message": "no"}))
    assert gh.create_pull_request("T", "B", "h", "main", ["a"])["number"] == 9


def test_errors_raise_with_status_and_message():
    gh, _ = client((403, {"message": "Resource not accessible by integration"}))
    with pytest.raises(GitHubError) as exc:
        gh.create_issue("T", "B", [])
    assert exc.value.status == 403 and "not accessible" in exc.value.message


def test_ensure_label_ignores_errors():
    gh, _ = client((422, {"message": "already_exists"}))
    gh.ensure_label("x", "ffffff", "d")  # must not raise

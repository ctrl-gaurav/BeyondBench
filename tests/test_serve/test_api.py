"""
Comprehensive tests for BeyondBench FastAPI server (Phase 11).

Tests cover:
- Health endpoint (version, gpu_count, uptime)
- Task listing and detail
- Evaluation job submission (single + batch)
- Job status, results, cancellation
- Result listing and detail
- Comparison API (POST /compare and POST /evaluate/compare)
- Prometheus metrics endpoint
- Authentication (Bearer token)
- Rate limiting
- WebSocket live progress streaming
- OpenAPI documentation availability
"""

import asyncio
import json
import os
import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# FastAPI test client — skip entire module if fastapi is not installed
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed (requires [serve] extra)")
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_server_state():
    """Reset in-memory server state between tests."""
    from beyondbench.serve import app as app_module

    app_module._jobs.clear()
    app_module._rate_state.clear()
    app_module._ws_subscribers.clear()
    app_module._GLOBAL_PROGRESS_QUEUES.clear()
    yield
    app_module._jobs.clear()
    app_module._rate_state.clear()


@pytest.fixture()
def app():
    from beyondbench.serve.app import create_app
    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def authed_client(app, monkeypatch):
    """Client with API key auth enabled."""
    from beyondbench.serve import app as app_module

    monkeypatch.setattr(app_module, "_API_KEY", "test-secret-key")
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_has_gpu_count(self, client):
        data = client.get("/health").json()
        assert "gpu_count" in data
        assert isinstance(data["gpu_count"], int)
        assert data["gpu_count"] >= 0

    def test_health_has_uptime(self, client):
        data = client.get("/health").json()
        assert "uptime" in data
        assert isinstance(data["uptime"], (int, float))
        assert data["uptime"] >= 0

    def test_health_no_auth_required(self, authed_client):
        """Health endpoint should work without auth even when auth is enabled."""
        resp = authed_client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_prometheus_format(self, client):
        resp = client.get("/metrics")
        text = resp.text
        assert "beyondbench_up 1" in text
        assert "beyondbench_uptime_seconds" in text
        assert "beyondbench_gpu_count" in text
        assert "beyondbench_jobs_total" in text
        assert "beyondbench_rate_limit_rpm" in text

    def test_metrics_has_help_and_type(self, client):
        text = client.get("/metrics").text
        assert "# HELP beyondbench_up" in text
        assert "# TYPE beyondbench_up gauge" in text

    def test_metrics_job_counts(self, client):
        """Metrics should reflect job state."""
        from beyondbench.serve import app as app_module

        # Add a fake completed job
        app_module._jobs["fake-1"] = {
            "status": "completed",
            "result": {},
            "error": None,
            "progress": None,
            "logs": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "cancel_flag": threading.Event(),
        }
        text = client.get("/metrics").text
        assert 'beyondbench_jobs_total{status="completed"} 1' in text
        assert "beyondbench_jobs_count 1" in text


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class TestTasks:
    def test_list_tasks(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert "total" in data
        assert data["total"] > 0

    def test_list_tasks_filter_easy(self, client):
        resp = client.get("/tasks?suite=easy")
        assert resp.status_code == 200
        data = resp.json()
        # Only easy tasks should be returned
        assert "easy" in data["tasks"]

    def test_get_task_info(self, client):
        resp = client.get("/tasks/sum")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"]["name"] == "sum"

    def test_get_task_not_found(self, client):
        resp = client.get("/tasks/nonexistent_task_xyz")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Evaluation Job Submission
# ---------------------------------------------------------------------------

class TestEvaluateSubmit:
    def _mock_eval_thread(self):
        """Patch _run_evaluation_thread to avoid real model loading."""
        def fake_thread(job_id, req):
            from beyondbench.serve import app as app_module
            with app_module._jobs_lock:
                job = app_module._jobs[job_id]
                job["status"] = "completed"
                job["result"] = {"summary": {"overall_accuracy": 0.5}}
                job["updated_at"] = datetime.utcnow()
        return patch("beyondbench.serve.app._run_evaluation_thread", side_effect=fake_thread)

    def test_submit_evaluation(self, client):
        with self._mock_eval_thread():
            resp = client.post("/evaluate", json={
                "model_id": "test-model",
                "backend": "vllm",
                "suite": "easy",
                "datapoints": 5,
            })
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "started"

    def test_submit_batch_evaluation(self, client):
        with self._mock_eval_thread():
            resp = client.post("/evaluate/batch", json={
                "jobs": [
                    {"model_id": "model-a", "backend": "vllm", "suite": "easy", "datapoints": 5},
                    {"model_id": "model-b", "backend": "vllm", "suite": "easy", "datapoints": 5},
                ]
            })
        assert resp.status_code == 202
        data = resp.json()
        assert data["total"] == 2
        assert len(data["job_ids"]) == 2

    def test_batch_max_20(self, client):
        """Batch should reject >20 jobs."""
        resp = client.post("/evaluate/batch", json={
            "jobs": [{"model_id": f"m{i}", "suite": "easy"} for i in range(21)]
        })
        assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# Job Status / Results / Cancellation
# ---------------------------------------------------------------------------

class TestJobManagement:
    def _add_job(self, status="completed", result=None, error=None):
        from beyondbench.serve import app as app_module
        job_id = f"test-job-{status}"
        app_module._jobs[job_id] = {
            "status": status,
            "result": result or {"summary": {"accuracy": 0.8}},
            "error": error,
            "progress": None,
            "logs": ["line1", "line2"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "cancel_flag": threading.Event(),
        }
        return job_id

    def test_get_job_status(self, client):
        job_id = self._add_job()
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_get_job_not_found(self, client):
        resp = client.get("/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_list_jobs(self, client):
        self._add_job("completed")
        self._add_job("running")
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_list_jobs_filter(self, client):
        self._add_job("completed")
        self._add_job("running")
        resp = client.get("/jobs?status=running")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        assert resp.json()["jobs"][0]["status"] == "running"

    def test_get_job_results_completed(self, client):
        job_id = self._add_job("completed", result={"summary": {"acc": 0.9}})
        resp = client.get(f"/jobs/{job_id}/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"]["summary"]["acc"] == 0.9

    def test_get_job_results_not_completed(self, client):
        job_id = self._add_job("running")
        resp = client.get(f"/jobs/{job_id}/results")
        assert resp.status_code == 409

    def test_get_job_results_not_found(self, client):
        resp = client.get("/jobs/nope/results")
        assert resp.status_code == 404

    def test_delete_running_job(self, client):
        job_id = self._add_job("running")
        resp = client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 200
        assert "cancel" in resp.json()["message"].lower() or "cancel" in resp.json()["message"]

    def test_delete_completed_job(self, client):
        job_id = self._add_job("completed")
        resp = client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 200
        # Should be removed from store
        resp2 = client.get(f"/jobs/{job_id}")
        assert resp2.status_code == 404


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class TestResults:
    def test_list_results_empty_dir(self, client, tmp_path):
        resp = client.get(f"/results?output_dir={tmp_path}")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_list_results_with_data(self, client, tmp_path):
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "final_results.json").write_text(
            json.dumps({"summary": {"accuracy": 0.75}})
        )
        resp = client.get(f"/results?output_dir={tmp_path}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["result_id"] == "run1"

    def test_get_result_detail(self, client, tmp_path):
        run_dir = tmp_path / "run_abc"
        run_dir.mkdir()
        payload = {"summary": {"accuracy": 0.88}, "task_results": {"sum": {"accuracy": 0.9}}}
        (run_dir / "final_results.json").write_text(json.dumps(payload))

        resp = client.get(f"/results/run_abc?output_dir={tmp_path}")
        assert resp.status_code == 200
        assert resp.json()["data"]["summary"]["accuracy"] == 0.88

    def test_get_result_not_found(self, client, tmp_path):
        resp = client.get(f"/results/nope?output_dir={tmp_path}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Comparison API
# ---------------------------------------------------------------------------

class TestCompare:
    def _setup_results(self, tmp_path):
        for name, acc in [("run_a", 0.7), ("run_b", 0.85)]:
            d = tmp_path / name
            d.mkdir()
            (d / "final_results.json").write_text(json.dumps({
                "summary": {"overall_accuracy": acc},
                "task_results": {"sum": {"accuracy": acc}},
            }))

    def test_compare_via_post_compare(self, client, tmp_path):
        self._setup_results(tmp_path)
        resp = client.post("/compare", json={
            "result_ids": ["run_a", "run_b"],
            "output_dir": str(tmp_path),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "run_a" in data["comparison"]
        assert "run_b" in data["comparison"]

    def test_compare_via_evaluate_compare(self, client, tmp_path):
        """POST /evaluate/compare should be an alias for POST /compare."""
        self._setup_results(tmp_path)
        resp = client.post("/evaluate/compare", json={
            "result_ids": ["run_a", "run_b"],
            "output_dir": str(tmp_path),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["result_ids"]) == 2

    def test_compare_with_metrics_filter(self, client, tmp_path):
        self._setup_results(tmp_path)
        resp = client.post("/compare", json={
            "result_ids": ["run_a", "run_b"],
            "output_dir": str(tmp_path),
            "metrics": ["overall_accuracy"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["comparison"]["run_a"]["overall_accuracy"] == 0.7

    def test_compare_not_found(self, client, tmp_path):
        resp = client.post("/compare", json={
            "result_ids": ["no_such", "also_missing"],
            "output_dir": str(tmp_path),
        })
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_no_auth_when_key_not_set(self, client):
        """All endpoints should work without auth when BEYONDBENCH_API_KEY is unset."""
        resp = client.get("/tasks")
        assert resp.status_code == 200

    def test_auth_required_when_key_set(self, authed_client):
        """Should 401 without Bearer token when key is configured."""
        resp = authed_client.get("/tasks")
        assert resp.status_code == 401

    def test_auth_with_valid_key(self, authed_client):
        resp = authed_client.get(
            "/tasks",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 200

    def test_auth_with_invalid_key(self, authed_client):
        resp = authed_client.get(
            "/tasks",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_auth_missing_bearer_prefix(self, authed_client):
        resp = authed_client.get(
            "/tasks",
            headers={"Authorization": "test-secret-key"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_rate_limit_enforced(self, app, monkeypatch):
        from beyondbench.serve import app as app_module
        monkeypatch.setattr(app_module, "_RATE_LIMIT_RPM", 3)
        client = TestClient(app)

        for i in range(3):
            resp = client.get("/tasks")
            assert resp.status_code == 200, f"Request {i+1} should succeed"

        resp = client.get("/tasks")
        assert resp.status_code == 429
        assert "rate limit" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# WebSocket — /ws/jobs/{job_id}
# ---------------------------------------------------------------------------

class TestWebSocketJobStream:
    def test_ws_finished_job_replays(self, app):
        """WS to a completed job should replay logs + completed event + done."""
        from beyondbench.serve import app as app_module

        job_id = "ws-test-done"
        app_module._jobs[job_id] = {
            "status": "completed",
            "result": {"summary": {"acc": 1.0}},
            "error": None,
            "progress": None,
            "logs": ["log line 1"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "cancel_flag": threading.Event(),
        }

        client = TestClient(app)
        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            msg1 = ws.receive_json()
            assert msg1["type"] == "log"
            msg2 = ws.receive_json()
            assert msg2["type"] == "completed"
            msg3 = ws.receive_json()
            assert msg3["type"] == "done"

    def test_ws_job_not_found(self, app):
        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/jobs/nonexistent"):
                pass

    def test_ws_auth_rejected(self, app, monkeypatch):
        from beyondbench.serve import app as app_module
        monkeypatch.setattr(app_module, "_API_KEY", "secret")

        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/jobs/some-id?api_key=wrong"):
                pass


# ---------------------------------------------------------------------------
# OpenAPI / Swagger
# ---------------------------------------------------------------------------

class TestOpenAPI:
    def test_docs_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_available(self, client):
        resp = client.get("/redoc")
        assert resp.status_code == 200

    def test_openapi_schema(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "BeyondBench API"
        # Check all expected paths exist
        paths = schema["paths"]
        assert "/health" in paths
        assert "/tasks" in paths
        assert "/evaluate" in paths
        assert "/jobs" in paths
        assert "/compare" in paths
        assert "/evaluate/compare" in paths
        assert "/results" in paths
        assert "/metrics" in paths

    def test_openapi_has_examples(self, client):
        schema = client.get("/openapi.json").json()
        # EvaluateRequest should have examples
        evaluate_schema = schema["paths"]["/evaluate"]["post"]
        assert "requestBody" in evaluate_schema

    def test_openapi_has_tags(self, client):
        schema = client.get("/openapi.json").json()
        tag_names = [t["name"] for t in schema.get("tags", [])]
        assert "System" in tag_names
        assert "Tasks" in tag_names
        assert "Evaluation" in tag_names
        assert "Results" in tag_names
        assert "Comparison" in tag_names


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_suite_rejected(self, client):
        resp = client.get("/tasks?suite=invalid")
        assert resp.status_code == 422

    def test_evaluate_missing_model_id(self, client):
        resp = client.post("/evaluate", json={"suite": "easy"})
        assert resp.status_code == 422

    def test_batch_empty_jobs_list(self, client):
        resp = client.post("/evaluate/batch", json={"jobs": []})
        assert resp.status_code == 422

    def test_compare_single_result_rejected(self, client, tmp_path):
        resp = client.post("/compare", json={
            "result_ids": ["only_one"],
            "output_dir": str(tmp_path),
        })
        assert resp.status_code == 422  # min_length=2

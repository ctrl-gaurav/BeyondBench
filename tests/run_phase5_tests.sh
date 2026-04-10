#!/usr/bin/env bash
# Phase 5 Test Runner — BeyondBench Comprehensive Test Suite
#
# Usage:
#   bash tests/run_phase5_tests.sh [--unit-only] [--skip-integration] [--skip-e2e]
#
# Requires:
#   - 8x A40 GPUs (or adapt CUDA_VISIBLE_DEVICES as needed)
#   - Models: Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct
#   - pip install -e .[dev]

set -euo pipefail

UNIT_ONLY=false
SKIP_INTEGRATION=false
SKIP_E2E=false

for arg in "$@"; do
  case $arg in
    --unit-only)       UNIT_ONLY=true ;;
    --skip-integration) SKIP_INTEGRATION=true ;;
    --skip-e2e)        SKIP_E2E=true ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "======================================================================"
echo "BeyondBench Phase 5 — Comprehensive Test Suite"
echo "======================================================================"

# ---------------------------------------------------------------------------
# 1. Unit tests (no GPU, fast)
# ---------------------------------------------------------------------------
echo ""
echo ">>> [1/3] Unit tests (no GPU required)..."
python -m pytest tests/unit/ -v --timeout=120 -x 2>&1 | tee /tmp/bb_unit_test.log
echo ""
echo "Unit test log: /tmp/bb_unit_test.log"

if $UNIT_ONLY; then
  echo "Unit-only mode — stopping here."
  exit 0
fi

if $SKIP_INTEGRATION; then
  echo "Skipping integration tests."
else
  # ---------------------------------------------------------------------------
  # 2. Integration tests (real GPUs, parallel)
  # ---------------------------------------------------------------------------
  echo ""
  echo ">>> [2/3] Integration tests (real models, parallel across GPUs)..."

  CUDA_VISIBLE_DEVICES=0 python -m pytest \
    tests/integration/test_easy_tasks_real_model.py \
    -v --timeout=600 2>&1 | tee /tmp/bb_easy_integration.log &
  PID_EASY=$!

  CUDA_VISIBLE_DEVICES=1 python -m pytest \
    tests/integration/test_medium_tasks_real_model.py \
    -v --timeout=600 2>&1 | tee /tmp/bb_medium_integration.log &
  PID_MEDIUM=$!

  CUDA_VISIBLE_DEVICES=2 python -m pytest \
    tests/integration/test_hard_tasks_real_model.py \
    -v --timeout=600 2>&1 | tee /tmp/bb_hard_integration.log &
  PID_HARD=$!

  CUDA_VISIBLE_DEVICES=3 python -m pytest \
    tests/integration/test_multi_model.py \
    -v --timeout=600 2>&1 | tee /tmp/bb_multi_model.log &
  PID_MULTI=$!

  # API backends — only if keys available
  python -m pytest tests/integration/test_api_backends.py \
    -v --timeout=300 2>&1 | tee /tmp/bb_api_backends.log &
  PID_API=$!

  echo "Waiting for all integration tests to complete..."
  wait $PID_EASY $PID_MEDIUM $PID_HARD $PID_MULTI $PID_API

  echo ""
  echo "Integration test logs:"
  echo "  Easy:         /tmp/bb_easy_integration.log"
  echo "  Medium:       /tmp/bb_medium_integration.log"
  echo "  Hard:         /tmp/bb_hard_integration.log"
  echo "  Multi-model:  /tmp/bb_multi_model.log"
  echo "  API backends: /tmp/bb_api_backends.log"
fi

if $SKIP_E2E; then
  echo "Skipping E2E tests."
else
  # ---------------------------------------------------------------------------
  # 3. E2E tests
  # ---------------------------------------------------------------------------
  echo ""
  echo ">>> [3/3] E2E tests..."
  CUDA_VISIBLE_DEVICES=7 python -m pytest \
    tests/e2e/ \
    -v --timeout=1200 2>&1 | tee /tmp/bb_e2e.log
  echo ""
  echo "E2E test log: /tmp/bb_e2e.log"
fi

echo ""
echo "======================================================================"
echo "Phase 5 test run complete."
echo "======================================================================"

# Summary
echo ""
echo "=== Summary ==="
grep -E "passed|failed|error" /tmp/bb_unit_test.log | tail -1 || true
grep -E "passed|failed|error" /tmp/bb_easy_integration.log 2>/dev/null | tail -1 || true
grep -E "passed|failed|error" /tmp/bb_medium_integration.log 2>/dev/null | tail -1 || true
grep -E "passed|failed|error" /tmp/bb_hard_integration.log 2>/dev/null | tail -1 || true
grep -E "passed|failed|error" /tmp/bb_e2e.log 2>/dev/null | tail -1 || true

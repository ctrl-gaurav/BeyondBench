#!/bin/bash
# =============================================================================
# BeyondBench Unified Evaluation Script
# Supports: OpenAI, Gemini, Anthropic, vLLM, and Transformers backends
# =============================================================================

set -e

# --- DISPLAY HELP ---
show_help() {
    echo "BeyondBench Unified Evaluation Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Backend Selection (choose one):"
    echo "  --backend TYPE        Local backend: 'vllm' (default) or 'transformers'"
    echo "  --api-provider PROV   API backend: 'openai', 'gemini', or 'anthropic'"
    echo ""
    echo "Model Configuration:"
    echo "  --model-id MODEL      Model identifier (required)"
    echo "  --api-key KEY         API key (or set via environment variable)"
    echo ""
    echo "Task Selection:"
    echo "  --suite SUITE         Task suite: 'easy', 'medium', 'hard', or 'all' (default: all)"
    echo "  --tasks TASKS         Specific tasks (comma-separated)"
    echo ""
    echo "Evaluation Parameters:"
    echo "  --datapoints N        Number of data points per task (default: 50)"
    echo "  --temperature T       Sampling temperature (default: 0.1)"
    echo "  --max-tokens N        Maximum tokens to generate (default: 32768)"
    echo "  --seed SEED           Random seed (default: 42)"
    echo ""
    echo "API-Specific Options:"
    echo "  --reasoning-effort E  OpenAI GPT-5 reasoning: minimal, low, medium, high"
    echo "  --thinking-budget N   Gemini thinking budget (0 to disable, -1 for dynamic)"
    echo ""
    echo "Local Model Options:"
    echo "  --tensor-parallel N   Number of GPUs for tensor parallelism (default: 1)"
    echo "  --gpu-memory-util F   GPU memory utilization 0.0-1.0 (default: 0.96)"
    echo "  --trust-remote-code   Trust remote code from HuggingFace"
    echo ""
    echo "Output Options:"
    echo "  --output-dir DIR      Output directory (default: ./results)"
    echo "  --store-details       Store detailed per-example results"
    echo ""
    echo "Examples:"
    echo "  # Evaluate GPT-4o on easy suite"
    echo "  $0 --api-provider openai --model-id gpt-4o --suite easy"
    echo ""
    echo "  # Evaluate local model with vLLM"
    echo "  $0 --backend vllm --model-id Qwen/Qwen2.5-3B-Instruct --suite all"
    echo ""
    echo "  # Evaluate Claude on hard tasks"
    echo "  $0 --api-provider anthropic --model-id claude-sonnet-4-20250514 --suite hard"
    echo ""
    exit 0
}

# --- DEFAULT CONFIGURATION ---
BACKEND="vllm"
API_PROVIDER=""
MODEL_ID=""
API_KEY=""
SUITE="all"
TASKS=""
DATAPOINTS=50
TEMPERATURE=0.1
TOP_P=0.9
MAX_TOKENS=32768
SEED=42
OUTPUT_DIR="./results"
STORE_DETAILS=""
REASONING_EFFORT=""
THINKING_BUDGET=""
TENSOR_PARALLEL=1
GPU_MEMORY_UTIL=0.96
TRUST_REMOTE_CODE=""

# --- PARSE ARGUMENTS ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --api-provider)
            API_PROVIDER="$2"
            shift 2
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --suite)
            SUITE="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --datapoints)
            DATAPOINTS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --store-details)
            STORE_DETAILS="--store-details"
            shift
            ;;
        --reasoning-effort)
            REASONING_EFFORT="$2"
            shift 2
            ;;
        --thinking-budget)
            THINKING_BUDGET="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --gpu-memory-util)
            GPU_MEMORY_UTIL="$2"
            shift 2
            ;;
        --trust-remote-code)
            TRUST_REMOTE_CODE="--trust-remote-code"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# --- VALIDATION ---
if [[ -z "$MODEL_ID" ]]; then
    echo "Error: --model-id is required"
    echo "Use --help for usage information"
    exit 1
fi

# --- API KEY HANDLING ---
if [[ -n "$API_PROVIDER" ]]; then
    case $API_PROVIDER in
        openai)
            API_KEY="${API_KEY:-$OPENAI_API_KEY}"
            if [[ -z "$API_KEY" ]]; then
                echo "Error: OpenAI API key required. Set --api-key or OPENAI_API_KEY environment variable"
                exit 1
            fi
            ;;
        gemini)
            API_KEY="${API_KEY:-$GEMINI_API_KEY}"
            if [[ -z "$API_KEY" ]]; then
                echo "Error: Gemini API key required. Set --api-key or GEMINI_API_KEY environment variable"
                exit 1
            fi
            ;;
        anthropic)
            API_KEY="${API_KEY:-$ANTHROPIC_API_KEY}"
            if [[ -z "$API_KEY" ]]; then
                echo "Error: Anthropic API key required. Set --api-key or ANTHROPIC_API_KEY environment variable"
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown API provider: $API_PROVIDER"
            echo "Supported providers: openai, gemini, anthropic"
            exit 1
            ;;
    esac
fi

# --- DISPLAY CONFIGURATION ---
echo "========================================================================"
echo "BeyondBench Evaluation"
echo "========================================================================"
echo "Model: $MODEL_ID"
if [[ -n "$API_PROVIDER" ]]; then
    echo "Backend: API ($API_PROVIDER)"
else
    echo "Backend: Local ($BACKEND)"
fi
echo "Suite: $SUITE"
echo "Datapoints: $DATAPOINTS"
echo "Temperature: $TEMPERATURE"
echo "Max Tokens: $MAX_TOKENS"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "------------------------------------------------------------------------"

# --- BUILD COMMAND ---
CMD="beyondbench evaluate --model-id \"$MODEL_ID\" --suite $SUITE --datapoints $DATAPOINTS --temperature $TEMPERATURE --top-p $TOP_P --max-tokens $MAX_TOKENS --seed $SEED --output-dir \"$OUTPUT_DIR\""

# Add backend/API options
if [[ -n "$API_PROVIDER" ]]; then
    CMD="$CMD --api-provider $API_PROVIDER --api-key \"$API_KEY\""

    # Add API-specific options
    if [[ -n "$REASONING_EFFORT" ]]; then
        CMD="$CMD --reasoning-effort $REASONING_EFFORT"
    fi
    if [[ -n "$THINKING_BUDGET" ]]; then
        CMD="$CMD --thinking-budget $THINKING_BUDGET"
    fi
else
    CMD="$CMD --backend $BACKEND"
    CMD="$CMD --tensor-parallel-size $TENSOR_PARALLEL"
    CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTIL"
    if [[ -n "$TRUST_REMOTE_CODE" ]]; then
        CMD="$CMD $TRUST_REMOTE_CODE"
    fi
fi

# Add specific tasks if provided
if [[ -n "$TASKS" ]]; then
    CMD="$CMD --tasks $TASKS"
fi

# Add store details flag
if [[ -n "$STORE_DETAILS" ]]; then
    CMD="$CMD $STORE_DETAILS"
fi

# --- EXECUTE ---
echo "Executing: $CMD"
echo "------------------------------------------------------------------------"

START_TIME=$(date +%s)

eval $CMD
EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "------------------------------------------------------------------------"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi
echo "Duration: ${DURATION}s"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================================================"

exit $EXIT_CODE

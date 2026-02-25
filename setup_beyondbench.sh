#!/bin/bash
# =============================================================================
# BeyondBench Interactive Setup Script
# Creates conda environment, installs dependencies, and verifies installation
# =============================================================================

set -e

# --- ANSI Color Codes ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Configuration ---
ENV_NAME="${BEYONDBENCH_ENV:-beyondbench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# --- Animation Functions ---
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " ${CYAN}[%c]${NC}  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "\r${CYAN}["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "]${NC} ${WHITE}%3d%%${NC}" $percentage
}

# --- Print Functions ---
print_banner() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║    ██████╗ ███████╗██╗   ██╗ ██████╗ ███╗   ██╗██████╗                   ║
    ║    ██╔══██╗██╔════╝╚██╗ ██╔╝██╔═══██╗████╗  ██║██╔══██╗                  ║
    ║    ██████╔╝█████╗   ╚████╔╝ ██║   ██║██╔██╗ ██║██║  ██║                  ║
    ║    ██╔══██╗██╔══╝    ╚██╔╝  ██║   ██║██║╚██╗██║██║  ██║                  ║
    ║    ██████╔╝███████╗   ██║   ╚██████╔╝██║ ╚████║██████╔╝                  ║
    ║    ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═════╝                   ║
    ║                                                                          ║
    ║    ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗                            ║
    ║    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║                            ║
    ║    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║                            ║
    ║    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║                            ║
    ║    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║                            ║
    ║    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝                            ║
    ║                                                                          ║
    ║              Interactive Setup & Installation Wizard                     ║
    ║                          Version 1.0.0                                   ║
    ╚══════════════════════════════════════════════════════════════════════════╝

EOF
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${WHITE}${BOLD}  STEP $1: $2${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}  ✅ $1${NC}"
}

print_error() {
    echo -e "${RED}  ❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}  ⚠️  $1${NC}"
}

print_info() {
    echo -e "${CYAN}  ℹ️  $1${NC}"
}

# --- Check Prerequisites ---
check_prerequisites() {
    print_step "1" "Checking Prerequisites"

    local all_good=true

    # Check for conda
    echo -e "  ${WHITE}Checking for conda...${NC}"
    if command -v conda &> /dev/null; then
        local conda_version=$(conda --version 2>&1 | head -1)
        print_success "conda found: $conda_version"
    else
        print_error "conda not found!"
        print_info "Please install Miniconda or Anaconda first"
        print_info "Visit: https://docs.conda.io/en/latest/miniconda.html"
        all_good=false
    fi

    # Check for git
    echo -e "  ${WHITE}Checking for git...${NC}"
    if command -v git &> /dev/null; then
        local git_version=$(git --version 2>&1 | head -1)
        print_success "git found: $git_version"
    else
        print_warning "git not found (optional but recommended)"
    fi

    # Check for CUDA (optional)
    echo -e "  ${WHITE}Checking for CUDA...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        local cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        print_success "NVIDIA GPU found (Driver: $cuda_version)"
        GPU_AVAILABLE=true
    else
        print_warning "No NVIDIA GPU detected (vLLM will not be available)"
        GPU_AVAILABLE=false
    fi

    if [ "$all_good" = false ]; then
        echo -e "\n${RED}  Please install missing prerequisites and try again.${NC}"
        exit 1
    fi

    print_success "All prerequisites satisfied!"
}

# --- Create Conda Environment ---
create_conda_env() {
    print_step "2" "Creating Conda Environment"

    # Check if environment already exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "Environment '${ENV_NAME}' already exists"
        echo -e "  ${WHITE}Do you want to:${NC}"
        echo -e "    ${CYAN}1)${NC} Use existing environment"
        echo -e "    ${CYAN}2)${NC} Delete and recreate"
        echo -e "    ${CYAN}3)${NC} Exit"
        read -p "  Choose [1/2/3]: " choice

        case $choice in
            1)
                print_info "Using existing environment"
                ;;
            2)
                print_info "Removing existing environment..."
                conda env remove -n ${ENV_NAME} -y
                print_info "Creating new environment..."
                conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
                ;;
            3)
                echo -e "\n${YELLOW}  Setup cancelled.${NC}"
                exit 0
                ;;
            *)
                print_error "Invalid choice"
                exit 1
                ;;
        esac
    else
        print_info "Creating new conda environment: ${ENV_NAME}"
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    fi

    print_success "Conda environment ready!"
}

# --- Activate Environment ---
activate_env() {
    print_step "3" "Activating Environment"

    # Source conda for shell
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}

    print_success "Environment '${ENV_NAME}' activated"
    print_info "Python: $(python --version 2>&1)"
    print_info "pip: $(pip --version 2>&1 | cut -d' ' -f1-2)"
}

# --- Install Dependencies ---
install_dependencies() {
    print_step "4" "Installing Dependencies"

    echo -e "  ${WHITE}Upgrading pip...${NC}"
    pip install --upgrade pip -q
    print_success "pip upgraded"

    echo -e "  ${WHITE}Installing core dependencies...${NC}"
    pip install torch torchvision torchaudio -q
    print_success "PyTorch installed"

    pip install transformers>=4.30.0 -q
    print_success "Transformers installed"

    pip install numpy pandas scipy -q
    print_success "NumPy, Pandas, SciPy installed"

    pip install tqdm rich click colorama -q
    print_success "CLI utilities installed"

    pip install tiktoken sentencepiece -q
    print_success "Tokenization libraries installed"

    echo -e "  ${WHITE}Installing API clients...${NC}"
    pip install openai>=1.0.0 -q
    print_success "OpenAI client installed"

    pip install google-generativeai>=0.3.0 -q
    print_success "Gemini client installed"

    pip install anthropic>=0.18.0 -q
    print_success "Anthropic client installed"

    # Install vLLM if GPU available
    if [ "$GPU_AVAILABLE" = true ]; then
        echo -e "  ${WHITE}Installing vLLM (GPU acceleration)...${NC}"
        pip install vllm>=0.4.0 -q 2>/dev/null || print_warning "vLLM installation may require manual setup"
        print_success "vLLM installed"
    else
        print_warning "Skipping vLLM (no GPU detected)"
    fi

    pip install pytest pytest-cov pyyaml jsonschema -q
    print_success "Testing and configuration libraries installed"

    print_success "All dependencies installed!"
}

# --- Install BeyondBench Package ---
install_package() {
    print_step "5" "Installing BeyondBench Package"

    # Get the script's directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

    echo -e "  ${WHITE}Installing BeyondBench in development mode...${NC}"
    cd "$SCRIPT_DIR"
    pip install -e . -q

    print_success "BeyondBench package installed!"
    print_info "Package location: $SCRIPT_DIR"
}

# --- Verify Installation ---
verify_installation() {
    print_step "6" "Verifying Installation"

    local all_good=true

    echo -e "  ${WHITE}Testing core imports...${NC}"

    # Test imports
    python -c "import beyondbench" 2>/dev/null && print_success "beyondbench" || { print_error "beyondbench import failed"; all_good=false; }
    python -c "from beyondbench import EvaluationEngine" 2>/dev/null && print_success "EvaluationEngine" || { print_error "EvaluationEngine import failed"; all_good=false; }
    python -c "from beyondbench import TaskRegistry" 2>/dev/null && print_success "TaskRegistry" || { print_error "TaskRegistry import failed"; all_good=false; }
    python -c "from beyondbench import ModelHandler" 2>/dev/null && print_success "ModelHandler" || { print_error "ModelHandler import failed"; all_good=false; }

    echo -e "\n  ${WHITE}Testing dependencies...${NC}"
    python -c "import torch" 2>/dev/null && print_success "torch" || { print_error "torch import failed"; all_good=false; }
    python -c "import transformers" 2>/dev/null && print_success "transformers" || { print_error "transformers import failed"; all_good=false; }
    python -c "import numpy" 2>/dev/null && print_success "numpy" || { print_error "numpy import failed"; all_good=false; }
    python -c "import pandas" 2>/dev/null && print_success "pandas" || { print_error "pandas import failed"; all_good=false; }
    python -c "import rich" 2>/dev/null && print_success "rich" || { print_error "rich import failed"; all_good=false; }
    python -c "import click" 2>/dev/null && print_success "click" || { print_error "click import failed"; all_good=false; }
    python -c "import tiktoken" 2>/dev/null && print_success "tiktoken" || { print_error "tiktoken import failed"; all_good=false; }

    echo -e "\n  ${WHITE}Testing API clients...${NC}"
    python -c "import openai" 2>/dev/null && print_success "openai" || { print_warning "openai not available"; }
    python -c "import google.generativeai" 2>/dev/null && print_success "google-generativeai" || { print_warning "google-generativeai not available"; }
    python -c "import anthropic" 2>/dev/null && print_success "anthropic" || { print_warning "anthropic not available"; }

    if [ "$GPU_AVAILABLE" = true ]; then
        echo -e "\n  ${WHITE}Testing vLLM...${NC}"
        python -c "import vllm" 2>/dev/null && print_success "vllm" || { print_warning "vllm not available"; }
    fi

    echo -e "\n  ${WHITE}Testing CLI...${NC}"
    if command -v beyondbench &> /dev/null; then
        print_success "beyondbench CLI available"
    else
        print_warning "beyondbench CLI not found in PATH"
    fi

    if command -v beyondbench &> /dev/null; then
        print_success "beyondbench CLI available"
    else
        print_warning "beyondbench CLI not found in PATH"
    fi

    if [ "$all_good" = false ]; then
        print_warning "Some imports failed - package may not work correctly"
    else
        print_success "All imports verified successfully!"
    fi
}

# --- Run Tests ---
run_tests() {
    print_step "7" "Running Tests"

    echo -e "  ${WHITE}Do you want to run the test suite?${NC}"
    echo -e "    ${CYAN}1)${NC} Run quick tests"
    echo -e "    ${CYAN}2)${NC} Run full tests"
    echo -e "    ${CYAN}3)${NC} Skip tests"
    read -p "  Choose [1/2/3]: " choice

    case $choice in
        1)
            print_info "Running quick tests..."
            python -m pytest tests/ -v --tb=short -x 2>/dev/null || print_warning "Some tests failed or no tests found"
            ;;
        2)
            print_info "Running full test suite..."
            python -m pytest tests/ -v --cov=beyondbench 2>/dev/null || print_warning "Some tests failed or no tests found"
            ;;
        3)
            print_info "Skipping tests"
            ;;
        *)
            print_info "Skipping tests"
            ;;
    esac
}

# --- Test Model Backends ---
test_backends() {
    print_step "8" "Testing Model Backends (Optional)"

    echo -e "  ${WHITE}Do you want to test model backends?${NC}"
    echo -e "    ${CYAN}1)${NC} Yes, test with a small model"
    echo -e "    ${CYAN}2)${NC} No, skip this step"
    read -p "  Choose [1/2]: " choice

    if [ "$choice" = "1" ]; then
        echo -e "\n  ${WHITE}Testing Transformers backend...${NC}"
        python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print('  ✅ Transformers backend working')
except Exception as e:
    print(f'  ⚠️  Transformers test failed: {e}')
" 2>/dev/null

        if [ "$GPU_AVAILABLE" = true ]; then
            echo -e "\n  ${WHITE}Testing PyTorch CUDA...${NC}"
            python -c "
import torch
if torch.cuda.is_available():
    print(f'  ✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'  ✅ CUDA version: {torch.version.cuda}')
else:
    print('  ⚠️  CUDA not available in PyTorch')
" 2>/dev/null
        fi
    else
        print_info "Skipping backend tests"
    fi
}

# --- Print Summary ---
print_summary() {
    echo -e "\n${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                          ║"
    echo "║              🎉 BeyondBench Installation Complete! 🎉                   ║"
    echo "║                                                                          ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    echo -e "${WHITE}  Quick Start Guide:${NC}"
    echo -e "  ─────────────────────────────────────────────────────────────────────────"
    echo -e "  ${CYAN}1. Activate environment:${NC}"
    echo -e "     conda activate ${ENV_NAME}"
    echo ""
    echo -e "  ${CYAN}2. Run interactive wizard:${NC}"
    echo -e "     beyondbench"
    echo ""
    echo -e "  ${CYAN}3. Run evaluation:${NC}"
    echo -e "     beyondbench evaluate --model-id gpt-4o --api-provider openai --suite easy"
    echo ""
    echo -e "  ${CYAN}4. List available tasks:${NC}"
    echo -e "     beyondbench list-tasks"
    echo ""
    echo -e "  ${WHITE}Environment Variables (set before running):${NC}"
    echo -e "  ─────────────────────────────────────────────────────────────────────────"
    echo -e "  ${YELLOW}export OPENAI_API_KEY=\"your-key\"${NC}      # For OpenAI models"
    echo -e "  ${YELLOW}export GEMINI_API_KEY=\"your-key\"${NC}      # For Gemini models"
    echo -e "  ${YELLOW}export ANTHROPIC_API_KEY=\"your-key\"${NC}   # For Claude models"
    echo ""
    echo -e "  ${WHITE}Documentation:${NC}"
    echo -e "  ─────────────────────────────────────────────────────────────────────────"
    echo -e "  README: ${CYAN}$(pwd)/README.md${NC}"
    echo -e "  Paper:  ${CYAN}$(pwd)/main.tex${NC}"
    echo ""
    echo -e "${GREEN}  Happy Evaluating! 🚀${NC}\n"
}

# --- Main Execution ---
main() {
    print_banner

    echo -e "${WHITE}  Welcome to the BeyondBench Setup Wizard!${NC}"
    echo -e "  This script will guide you through the installation process."
    echo ""
    echo -e "  ${CYAN}Configuration:${NC}"
    echo -e "    • Environment name: ${WHITE}${ENV_NAME}${NC}"
    echo -e "    • Python version: ${WHITE}${PYTHON_VERSION}${NC}"
    echo ""
    read -p "  Press ENTER to continue or Ctrl+C to cancel..."

    check_prerequisites
    create_conda_env
    activate_env
    install_dependencies
    install_package
    verify_installation
    run_tests
    test_backends
    print_summary
}

# Run main function
main

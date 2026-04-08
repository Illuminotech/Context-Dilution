#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3.11}"
RESULTS_DIR="results"
LOG_FILE="experiment.log"

usage() {
    cat <<EOF
Usage: ./run.sh <command> [options]

Commands:
  setup          Install dependencies and pull Ollama models
  pilot          Run N=1 pilot (~2-3 hours on M1 Max)
  run [N]        Run experiment with N trials per cell (default: 15)
  single TASK    Run a single task for debugging
  evaluate       Run blinded human evaluation on completed results
  analyze        Re-run analysis and regenerate report
  status         Show experiment progress
  clean          Clear all results

Options:
  -b, --background    Run experiment in the background
  -c, --condition     Context condition for single task (default: full)
  -v, --verbose       Verbose output

Examples:
  ./run.sh setup
  ./run.sh pilot
  ./run.sh pilot --background
  ./run.sh run 2
  ./run.sh single sequential_debug_001
  ./run.sh single sequential_debug_001 --condition minimal
  ./run.sh evaluate
  ./run.sh analyze
  ./run.sh status
  ./run.sh clean
EOF
    exit 1
}

check_ollama() {
    if ! curl -s http://localhost:11434/api/ps > /dev/null 2>&1; then
        echo "Error: Ollama is not running. Start the Ollama app first."
        exit 1
    fi
}

check_results() {
    if [ ! -d "$RESULTS_DIR/raw" ] || [ -z "$(ls -A "$RESULTS_DIR/raw" 2>/dev/null)" ]; then
        echo "Error: No results found. Run the experiment first."
        exit 1
    fi
}

cmd_setup() {
    echo "Installing Python dependencies..."
    $PYTHON -m pip install -e ".[dev]"

    echo ""
    echo "Pulling Ollama models..."
    ollama pull qwen2.5-coder:14b
    ollama pull gpt-oss:20b
    ollama pull llama3.2

    echo ""
    echo "Verifying Ollama GPU..."
    curl -s http://localhost:11434/api/generate -d '{"model":"llama3.2","prompt":"hi","options":{"num_predict":1}}' > /dev/null 2>&1
    VRAM=$(curl -s http://localhost:11434/api/ps | $PYTHON -c "import json,sys; d=json.load(sys.stdin); print(sum(m.get('size_vram',0) for m in d.get('models',[])))" 2>/dev/null || echo "0")
    if [ "$VRAM" = "0" ]; then
        echo "Warning: No GPU/VRAM detected. Inference will be very slow."
        echo "Make sure you're using the Ollama .app, not the Homebrew version."
    else
        echo "GPU active: $(echo "$VRAM" | $PYTHON -c "import sys; print(f'{int(sys.stdin.read())/1e9:.1f}GB VRAM')")"
    fi

    echo ""
    echo "Setup complete. Run './run.sh pilot' to start."
}

cmd_experiment() {
    local trials="${1:-15}"
    local background=false
    local verbose=""
    shift || true

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -b|--background) background=true; shift ;;
            -v|--verbose) verbose="-v"; shift ;;
            *) shift ;;
        esac
    done

    check_ollama

    if [ -d "$RESULTS_DIR/raw" ] && [ -n "$(ls -A "$RESULTS_DIR/raw" 2>/dev/null)" ]; then
        echo "Warning: Previous results exist in $RESULTS_DIR/"
        read -p "Clear and start fresh? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$RESULTS_DIR"
        else
            echo "Aborting. Run './run.sh clean' first or remove results manually."
            exit 1
        fi
    fi

    if $background; then
        echo "Starting experiment in background (N=$trials)..."
        nohup $PYTHON -m scripts.run_experiment --trials "$trials" $verbose > "$LOG_FILE" 2>&1 &
        local pid=$!
        echo "PID: $pid"
        echo "$pid" > .experiment.pid
        echo "Log: $LOG_FILE"
        echo ""
        echo "Monitor with:  ./run.sh status"
        echo "Or:            tail -f $LOG_FILE | grep 'score='"
        echo "Kill with:     kill $pid"
    else
        $PYTHON -m scripts.run_experiment --trials "$trials" $verbose
    fi
}

cmd_single() {
    local task_id="${1:?Error: task ID required. Example: ./run.sh single sequential_debug_001}"
    local condition="full"
    local verbose=""
    shift

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--condition) condition="$2"; shift 2 ;;
            -v|--verbose) verbose="-v"; shift ;;
            *) shift ;;
        esac
    done

    check_ollama
    $PYTHON -m scripts.run_single_task "$task_id" --condition "$condition" $verbose
}

cmd_evaluate() {
    check_results
    $PYTHON -m scripts.run_human_eval "$@"
}

cmd_analyze() {
    check_results
    $PYTHON -m scripts.analyze_results "$@"
    echo ""
    echo "Report: $RESULTS_DIR/report.md"
    echo "Figures: $RESULTS_DIR/figures/"
}

cmd_status() {
    if [ ! -d "$RESULTS_DIR/raw" ]; then
        echo "No experiment running or completed."
        exit 0
    fi

    local completed
    completed=$(ls "$RESULTS_DIR/raw" 2>/dev/null | wc -l | tr -d ' ')
    echo "Completed trials: $completed"

    if [ -f "$RESULTS_DIR/scored/all_trials.csv" ]; then
        echo "Status: Experiment finished"
        echo "Report: $RESULTS_DIR/report.md"
    elif [ -f .experiment.pid ] && kill -0 "$(cat .experiment.pid)" 2>/dev/null; then
        echo "Status: Running (PID $(cat .experiment.pid))"
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Last 5 scores:"
            grep "score=" "$LOG_FILE" 2>/dev/null | tail -5
        fi
    else
        echo "Status: Stopped (incomplete)"
    fi
}

cmd_clean() {
    if [ -d "$RESULTS_DIR" ]; then
        read -p "Delete all results in $RESULTS_DIR/? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$RESULTS_DIR"
            rm -f "$LOG_FILE" .experiment.pid
            echo "Cleaned."
        fi
    else
        echo "Nothing to clean."
    fi
}

# Parse command
command="${1:-}"
shift || true

case "$command" in
    setup)    cmd_setup ;;
    pilot)    cmd_experiment 1 "$@" ;;
    run)      cmd_experiment "$@" ;;
    single)   cmd_single "$@" ;;
    evaluate) cmd_evaluate "$@" ;;
    analyze)  cmd_analyze "$@" ;;
    status)   cmd_status ;;
    clean)    cmd_clean ;;
    *)        usage ;;
esac

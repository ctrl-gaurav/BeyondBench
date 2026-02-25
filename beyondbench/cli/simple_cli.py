#!/usr/bin/env python3
"""
Simplified CLI for beyondbench package that integrates with existing robust implementations.
"""

import sys
import os
import argparse
import logging

# Add package to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_easy_task(task_name, model_id, api_provider=None, api_key=None, **kwargs):
    """Run a single easy task."""
    try:
        # Import from your robust implementation
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'easy_suite', 'llmthinkbench'))

        from models.model_handler import ModelHandler
        from utils.logging_utils import setup_logging as robust_setup_logging

        # Setup output directory
        output_dir = kwargs.get('output_dir', './beyondbench_results')
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging using your robust implementation
        robust_setup_logging(output_dir)

        # Initialize model handler
        model_handler = ModelHandler(
            model_id=model_id,
            api_provider=api_provider,
            api_key=api_key,
            reasoning_effort=kwargs.get('reasoning_effort', 'medium'),
            thinking_budget=kwargs.get('thinking_budget', 1024)
        )

        print(f"✅ Initialized model handler for {model_id}")

        # Import specific task
        if task_name == "sorting":
            from tasks.sorting_task import SortingTask as TaskClass
        elif task_name == "comparison":
            from tasks.comparison_task import ComparisonTask as TaskClass
        elif task_name == "sum":
            from tasks.sum_task import SumTask as TaskClass
        else:
            print(f"❌ Task '{task_name}' not implemented yet")
            return False

        # Initialize task
        task = TaskClass(
            model_handler=model_handler,
            output_dir=output_dir,
            min_val=kwargs.get('range_min', -100),
            max_val=kwargs.get('range_max', 100),
            num_folds=kwargs.get('folds', 1),
            num_samples=kwargs.get('datapoints', 10),
            store_details=kwargs.get('store_details', False),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            max_tokens=kwargs.get('max_tokens', 512)
        )

        print(f"🎯 Running {task_name} evaluation...")

        # Run evaluation
        if hasattr(task, 'supports_list_sizes') and task.supports_list_sizes:
            list_sizes = kwargs.get('list_sizes', [8, 16])
            if isinstance(list_sizes, str):
                list_sizes = [int(x.strip()) for x in list_sizes.split(',')]
            results = task.run_evaluation(list_sizes)
        else:
            results = task.run_evaluation()

        print(f"🎉 Evaluation completed! Results saved to {output_dir}")

        # Print statistics
        if hasattr(model_handler, 'print_statistics_report'):
            model_handler.print_statistics_report()

        return True

    except Exception as e:
        print(f"❌ Error running task: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🔥 beyondbench: BeyondBench Evaluation Package (Simplified CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model configuration
    parser.add_argument('--model-id', required=True, help='Model identifier')
    parser.add_argument('--api-provider', choices=['openai', 'gemini'], help='API provider')
    parser.add_argument('--api-key', help='API key (or set environment variable)')

    # Task configuration
    parser.add_argument('--task', default='sorting', help='Task to run (default: sorting)')
    parser.add_argument('--datapoints', type=int, default=10, help='Number of datapoints (default: 10)')
    parser.add_argument('--folds', type=int, default=1, help='Number of folds (default: 1)')
    parser.add_argument('--list-sizes', default='8,16', help='List sizes for scalable tasks (default: 8,16)')

    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature (default: 0.7)')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (default: 0.9)')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens (default: 512)')

    # API-specific parameters
    parser.add_argument('--reasoning-effort', choices=['minimal', 'low', 'medium', 'high'],
                       default='medium', help='OpenAI reasoning effort (default: medium)')
    parser.add_argument('--thinking-budget', type=int, default=1024, help='Gemini thinking budget (default: 1024)')

    # Output configuration
    parser.add_argument('--output-dir', default='./beyondbench_results', help='Output directory')
    parser.add_argument('--store-details', action='store_true', help='Store detailed results')
    parser.add_argument('--range-min', type=int, default=-100, help='Min value range')
    parser.add_argument('--range-max', type=int, default=100, help='Max value range')

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Print banner
    print("🔥 beyondbench: BeyondBench Evaluation Package")
    print("=" * 50)

    # Get API key from environment if not provided
    if args.api_provider and not args.api_key:
        if args.api_provider == 'openai':
            args.api_key = os.getenv('OPENAI_API_KEY')
        elif args.api_provider == 'gemini':
            args.api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

        if not args.api_key:
            print(f"❌ API key required for {args.api_provider}. Set environment variable or use --api-key")
            return 1

    # Convert args to dict
    kwargs = vars(args)

    # Run evaluation
    success = run_easy_task(args.task, args.model_id, args.api_provider, args.api_key, **kwargs)

    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
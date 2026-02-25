#!/usr/bin/env python3
"""
Statistics Aggregator for GPT/Gemini Enhanced Medium Suite
Aggregates statistics from all task runs and provides comprehensive reporting.
"""

import json
import os
import sys
import time
import re
import glob
from pathlib import Path
import argparse

def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def count_tokens_fallback(text):
    """Fallback token counting method"""
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except:
        # Fallback to word-based estimation
        return int(len(text.split()) * 1.3)

def extract_statistics_from_logs(log_dir, model_id, api_provider):
    """Extract comprehensive statistics from log files"""
    print(f"📊 Extracting statistics from {log_dir}")

    stats = {
        'total_api_calls': 0,
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'tasks_completed': 0,
        'tasks_failed': 0,
        'api_errors': 0,
        'model_id': model_id,
        'api_provider': api_provider,
        'task_details': {}
    }

    log_files = glob.glob(f"{log_dir}/*.log")
    print(f"Found {len(log_files)} log files to analyze")

    for log_file in log_files:
        task_name = os.path.basename(log_file).replace('.log', '')
        print(f"  • Analyzing {task_name}")

        task_stats = {
            'api_calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'completed': False,
            'errors': 0
        }

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for successful completion using more comprehensive patterns
            success_patterns = [
                r'✅.*completed?',
                r'successfully completed',
                r'evaluation.*complete',
                r'task.*finished',
                r'results.*saved',
                r'acc=1\.000',  # Perfect accuracy
                r'100%.*acc=1\.000',  # Progress with perfect accuracy
                r'final_report\.json',  # JSON report generated
                r'All tasks completed!'  # Final success message
            ]

            # Also check for presence of results files
            task_name_clean = task_name.replace('_gpt-5-nano', '').replace('_gpt-4o-mini', '').replace('_openai', '').replace('_gemini', '')

            # Get to the root directory (go up from logs/model_tag to root)
            root_dir = os.path.dirname(os.path.dirname(log_dir))
            results_base_dir = os.path.join(root_dir, 'Results')

            # Extract model tag from log_dir path (e.g., logs/gpt-5-nano -> gpt-5-nano)
            model_tag = os.path.basename(log_dir)
            results_dir = os.path.join(results_base_dir, model_tag)

            # Look for corresponding results directory
            has_results = False
            if os.path.exists(results_dir):
                # First check for final_report.json in the main results directory
                main_final_report = os.path.join(results_dir, 'final_report.json')
                if os.path.exists(main_final_report):
                    has_results = True
                else:
                    # Check subdirectories
                    for item in os.listdir(results_dir):
                        if task_name_clean in item or any(part in item for part in task_name_clean.split('_')):
                            results_path = os.path.join(results_dir, item)
                            if os.path.isdir(results_path):
                                # Check if final_report.json exists
                                final_report = os.path.join(results_path, 'final_report.json')
                                if os.path.exists(final_report):
                                    has_results = True
                                    break

            if any(re.search(pattern, content, re.IGNORECASE) for pattern in success_patterns) or has_results:
                task_stats['completed'] = True
                stats['tasks_completed'] += 1
                print(f"    ✅ Task completed successfully")
            else:
                stats['tasks_failed'] += 1
                print(f"    ❌ Task appears to have failed")

            # Look for API-specific patterns with enhanced detection
            if api_provider == 'openai':
                # OpenAI specific patterns - look for actual HTTP requests
                api_call_patterns = [
                    r'HTTP Request: POST https://api\.openai\.com',
                    r'chat\.completions\.create',
                    r'openai.*api.*call',
                    r'response\.choices\[0\]',
                    r'INFO:httpx:HTTP Request.*openai\.com',
                    r'🤖 OpenAI API call:',  # Our new logging pattern
                    # Remove the problematic pattern that counts progress bar updates
                ]

                # Look for token usage in OpenAI response logs
                input_token_patterns = [
                    r'prompt_tokens=(\d+)',  # Our new logging format
                    r'prompt_tokens["\']?\s*:\s*(\d+)',
                    r'estimated_input_tokens=(\d+)',
                ]
                output_token_patterns = [
                    r'completion_tokens=(\d+)',  # Our new logging format
                    r'completion_tokens["\']?\s*:\s*(\d+)',
                    r'estimated_output_tokens=(\d+)',
                    r'"output_tokens":\s*\{[^}]*"mean":\s*(\d+\.?\d*)'  # From JSON results
                ]

            elif api_provider == 'gemini':
                # Gemini specific patterns
                api_call_patterns = [
                    r'HTTP Request.*googleapis\.com',
                    r'generate_content',
                    r'gemini.*api.*call',
                    r'GenerativeModel',
                    r'🧠 Gemini API call:',  # Our new logging pattern
                    # Remove the problematic pattern that counts progress bar updates
                ]

                input_token_patterns = [
                    r'estimated_input_tokens=(\d+)',  # Our new logging format
                    r'input.*tokens.*?(\d+)',
                ]
                output_token_patterns = [
                    r'estimated_output_tokens=(\d+)',  # Our new logging format
                    r'output.*tokens.*?(\d+)',
                    r'tokens.*generated.*?(\d+)',
                    r'"output_tokens":\s*\{[^}]*"mean":\s*(\d+\.?\d*)'
                ]
            else:
                # Generic patterns for any API
                api_call_patterns = [
                    r'api.*call',
                    r'generate.*request',
                    r'model.*response',
                    r'HTTP Request.*api\.',
                    # Remove the problematic pattern that counts progress bar updates
                ]
                # Generic patterns will be handled in the extraction logic below

            # Count API calls
            for pattern in api_call_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                task_stats['api_calls'] += len(matches)

            # Extract token counts from logs using separate input/output patterns
            if api_provider == 'openai' or api_provider == 'gemini':
                # Extract input tokens
                for pattern in input_token_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        try:
                            tokens = int(float(match))
                            task_stats['input_tokens'] += tokens
                        except ValueError:
                            continue

                # Extract output tokens
                for pattern in output_token_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        try:
                            tokens = int(float(match))
                            task_stats['output_tokens'] += tokens
                        except ValueError:
                            continue
            else:
                # Generic pattern extraction for other providers
                generic_token_patterns = [
                    r'tokens.*?(\d+)',
                    r'input.*?(\d+)',
                    r'output.*?(\d+)',
                    r'"output_tokens":\s*\{[^}]*"mean":\s*(\d+\.?\d*)'
                ]
                for pattern in generic_token_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        try:
                            tokens = int(float(match))
                            if 'input' in pattern or 'prompt' in pattern:
                                task_stats['input_tokens'] += tokens
                            elif 'output' in pattern or 'completion' in pattern:
                                task_stats['output_tokens'] += tokens
                        except ValueError:
                            continue

            # Enhanced: Read from JSON results files for accurate token counts
            if has_results:
                try:
                    final_report_path = None

                    # First check main results directory
                    main_final_report = os.path.join(results_dir, 'final_report.json')
                    if os.path.exists(main_final_report):
                        final_report_path = main_final_report
                    else:
                        # Check subdirectories
                        if os.path.exists(results_dir):
                            for item in os.listdir(results_dir):
                                if task_name_clean in item or any(part in item for part in task_name_clean.split('_')):
                                    results_dir_path = os.path.join(results_dir, item)
                                    if os.path.isdir(results_dir_path):
                                        potential_report = os.path.join(results_dir_path, 'final_report.json')
                                        if os.path.exists(potential_report):
                                            final_report_path = potential_report
                                            break

                    if final_report_path:
                        with open(final_report_path, 'r') as f:
                            report_data = json.load(f)

                            # Extract token usage from JSON results
                            if 'task_metrics' in report_data:
                                json_output_tokens = 0
                                api_calls_from_results = 0

                                for task_key, metrics in report_data['task_metrics'].items():
                                    # Skip average entries to avoid double counting
                                    if '_avg' in task_key:
                                        continue

                                    if 'output_tokens' in metrics and 'mean' in metrics['output_tokens']:
                                        tokens = int(float(metrics['output_tokens']['mean']))
                                        json_output_tokens += tokens
                                        api_calls_from_results += 1

                                if json_output_tokens > 0:
                                    # Prioritize JSON data as it's most accurate
                                    task_stats['output_tokens'] = json_output_tokens
                                    task_stats['api_calls'] = api_calls_from_results
                                    # Estimate input tokens based on typical prompt sizes for sequence tasks
                                    # Each test case typically has prompts of ~200-400 tokens
                                    estimated_input_tokens = api_calls_from_results * 300  # Conservative estimate
                                    if task_stats['input_tokens'] == 0:
                                        task_stats['input_tokens'] = estimated_input_tokens
                                    print(f"    📊 Extracted from JSON: {api_calls_from_results} API calls, {json_output_tokens} output tokens")

                except Exception as e:
                    print(f"    ⚠️ Could not read JSON results: {e}")

            # Fallback: estimate tokens from content length
            if task_stats['input_tokens'] == 0 and task_stats['output_tokens'] == 0:
                estimated_tokens = count_tokens_fallback(content)
                task_stats['input_tokens'] = estimated_tokens // 2
                task_stats['output_tokens'] = estimated_tokens // 2
                print(f"    📊 Estimated tokens: {estimated_tokens} (fallback method)")

            # Count errors
            error_patterns = [
                r'error',
                r'exception',
                r'failed',
                r'timeout',
                r'rate.*limit'
            ]

            for pattern in error_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                task_stats['errors'] += len(matches)

            if task_stats['errors'] > 0:
                stats['api_errors'] += task_stats['errors']

            print(f"    📞 API calls: {task_stats['api_calls']}")
            print(f"    📥 Input tokens: {task_stats['input_tokens']}")
            print(f"    📤 Output tokens: {task_stats['output_tokens']}")
            if task_stats['errors'] > 0:
                print(f"    ⚠️ Errors: {task_stats['errors']}")

        except Exception as e:
            print(f"    ❌ Error analyzing {log_file}: {e}")
            stats['tasks_failed'] += 1
            task_stats['errors'] += 1

        # Aggregate to global stats
        stats['total_api_calls'] += task_stats['api_calls']
        stats['total_input_tokens'] += task_stats['input_tokens']
        stats['total_output_tokens'] += task_stats['output_tokens']
        stats['task_details'][task_name] = task_stats

    return stats

def print_comprehensive_report(stats, start_time, end_time):
    """Print a comprehensive statistics report"""
    total_time = end_time - start_time
    total_tokens = stats['total_input_tokens'] + stats['total_output_tokens']

    # Calculate averages
    avg_input_tokens = stats['total_input_tokens'] / max(1, stats['total_api_calls'])
    avg_output_tokens = stats['total_output_tokens'] / max(1, stats['total_api_calls'])

    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RUN STATISTICS REPORT")
    print("="*80)
    print(f"🤖 Model: {stats['model_id']}")
    if stats['api_provider']:
        print(f"🌐 API Provider: {stats['api_provider'].upper()}")
    print(f"⏱️  Total Runtime: {format_duration(total_time)}")
    print(f"📞 Total API Calls: {stats['total_api_calls']:,}")
    print(f"✅ Tasks Completed: {stats['tasks_completed']}")
    if stats['tasks_failed'] > 0:
        print(f"❌ Tasks Failed: {stats['tasks_failed']}")
    if stats['api_errors'] > 0:
        print(f"⚠️ API Errors: {stats['api_errors']}")

    print("-"*80)
    print("📥 INPUT TOKENS:")
    print(f"   Total: {stats['total_input_tokens']:,}")
    print(f"   Average per call: {avg_input_tokens:.1f}")

    print("-"*80)
    print("📤 OUTPUT TOKENS:")
    print(f"   Total: {stats['total_output_tokens']:,}")
    print(f"   Average per call: {avg_output_tokens:.1f}")

    print("-"*80)
    print("📊 COMBINED METRICS:")
    print(f"   Total tokens: {total_tokens:,}")
    if total_time > 0:
        print(f"   Tokens per second: {total_tokens / total_time:.1f}")
        print(f"   API calls per second: {stats['total_api_calls'] / total_time:.2f}")

    print("-"*80)
    print("📋 TASK BREAKDOWN:")
    for task_name, task_stats in stats['task_details'].items():
        status = "✅" if task_stats['completed'] else "❌"
        print(f"   {status} {task_name}:")
        print(f"      API calls: {task_stats['api_calls']}")
        print(f"      Input tokens: {task_stats['input_tokens']:,}")
        print(f"      Output tokens: {task_stats['output_tokens']:,}")
        if task_stats['errors'] > 0:
            print(f"      Errors: {task_stats['errors']}")

    # Estimated costs for OpenAI
    if stats['api_provider'] == 'openai':
        print("-"*80)
        input_cost = output_cost = 0

        # Latest OpenAI pricing (per 1K tokens)
        if 'gpt-5' in stats['model_id'] and 'nano' in stats['model_id']:
            input_cost = stats['total_input_tokens'] * 0.05 / 1000  # $0.05 per 1K
            output_cost = stats['total_output_tokens'] * 0.40 / 1000  # $0.40 per 1K
        elif 'gpt-5' in stats['model_id'] and 'mini' in stats['model_id']:
            input_cost = stats['total_input_tokens'] * 0.25 / 1000  # $0.25 per 1K
            output_cost = stats['total_output_tokens'] * 2.00 / 1000  # $2.00 per 1K
        elif 'gpt-5' in stats['model_id'] and not ('mini' in stats['model_id'] or 'nano' in stats['model_id']):
            input_cost = stats['total_input_tokens'] * 1.25 / 1000  # $1.25 per 1K
            output_cost = stats['total_output_tokens'] * 10.00 / 1000  # $10.00 per 1K
        elif 'gpt-4.1' in stats['model_id'] and 'nano' in stats['model_id']:
            input_cost = stats['total_input_tokens'] * 0.10 / 1000  # $0.10 per 1K
            output_cost = stats['total_output_tokens'] * 0.40 / 1000  # $0.40 per 1K
        elif 'gpt-4.1' in stats['model_id'] and 'mini' in stats['model_id']:
            input_cost = stats['total_input_tokens'] * 0.40 / 1000  # $0.40 per 1K
            output_cost = stats['total_output_tokens'] * 1.60 / 1000  # $1.60 per 1K
        elif 'gpt-4.1' in stats['model_id']:
            input_cost = stats['total_input_tokens'] * 2.00 / 1000  # $2.00 per 1K
            output_cost = stats['total_output_tokens'] * 8.00 / 1000  # $8.00 per 1K
        elif 'gpt-4o-mini' in stats['model_id']:
            input_cost = stats['total_input_tokens'] * 0.15 / 1000  # $0.15 per 1K
            output_cost = stats['total_output_tokens'] * 0.60 / 1000  # $0.60 per 1K
        elif 'gpt-4o' in stats['model_id']:
            input_cost = stats['total_input_tokens'] * 2.50 / 1000  # $2.50 per 1K
            output_cost = stats['total_output_tokens'] * 10.00 / 1000  # $10.00 per 1K
        elif 'o4-mini' in stats['model_id']:
            input_cost = stats['total_input_tokens'] * 1.10 / 1000  # $1.10 per 1K
            output_cost = stats['total_output_tokens'] * 4.40 / 1000  # $4.40 per 1K
        elif 'o3-mini' in stats['model_id']:
            input_cost = stats['total_input_tokens'] * 1.10 / 1000  # $1.10 per 1K
            output_cost = stats['total_output_tokens'] * 4.40 / 1000  # $4.40 per 1K
        elif 'o3' in stats['model_id'] and not 'mini' in stats['model_id']:
            input_cost = stats['total_input_tokens'] * 2.00 / 1000  # $2.00 per 1K
            output_cost = stats['total_output_tokens'] * 8.00 / 1000  # $8.00 per 1K

        if input_cost > 0 or output_cost > 0:
            total_cost = input_cost + output_cost
            print(f"💰 ESTIMATED COST (OpenAI):")
            print(f"   Input tokens: ${input_cost:.4f}")
            print(f"   Output tokens: ${output_cost:.4f}")
            print(f"   Total: ${total_cost:.4f}")
            print("   (Note: Prices updated January 2025, check OpenAI pricing for current rates)")

    print("-"*80)
    print(f"🕐 Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"🕐 Finished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Aggregate statistics from Medium Suite evaluation runs')
    parser.add_argument('log_dir', help='Directory containing log files')
    parser.add_argument('--model_id', required=True, help='Model ID used in the run')
    parser.add_argument('--api_provider', help='API provider used (openai or gemini)')
    parser.add_argument('--start_time', type=int, required=True, help='Start time as Unix timestamp')
    parser.add_argument('--end_time', type=int, required=True, help='End time as Unix timestamp')

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"❌ Error: Log directory {args.log_dir} does not exist")
        sys.exit(1)

    # Extract statistics from logs
    stats = extract_statistics_from_logs(args.log_dir, args.model_id, args.api_provider)

    # Print comprehensive report
    print_comprehensive_report(stats, args.start_time, args.end_time)

    # Save stats to JSON file
    stats_file = os.path.join(args.log_dir, 'aggregated_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n📁 Detailed statistics saved to: {stats_file}")

if __name__ == "__main__":
    main()
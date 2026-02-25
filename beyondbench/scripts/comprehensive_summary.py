#!/usr/bin/env python3
"""
Comprehensive Summary Generator for GPT/Gemini Enhanced Easy Suite
"""

import json
import os
import sys
import glob
from pathlib import Path
import argparse

def load_results_from_directory(results_dir):
    """Load all result files from a results directory"""
    results = {}
    if not os.path.exists(results_dir):
        return results

    # First, look for the main final_report.json at the root level
    final_report_path = os.path.join(results_dir, 'final_report.json')
    if os.path.exists(final_report_path):
        try:
            with open(final_report_path, 'r') as f:
                data = json.load(f)
            # Use a descriptive name for the main report
            results['main_report'] = data
            return results  # Return early if we found the main report
        except Exception as e:
            print(f"⚠️ Could not load main final_report.json: {e}")

    # Fallback: Look for task-specific final reports in subdirectories
    for root, dirs, files in os.walk(results_dir):
        # Skip detailed results from test_case subdirectories
        if 'test_case_' in root or '.ipynb_checkpoints' in root:
            continue

        for file in files:
            if file.endswith('.json') and ('results' in file.lower() or 'final_report' in file.lower()):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Extract task name from directory structure
                    relative_path = os.path.relpath(root, results_dir)
                    if relative_path == '.':
                        task_name = os.path.splitext(file)[0]
                    else:
                        task_name = relative_path.split('/')[0]  # Use first directory as task name

                    results[task_name] = data
                except Exception as e:
                    print(f"⚠️ Could not load {file_path}: {e}")

    return results

def calculate_overall_metrics(all_results):
    """Calculate overall performance metrics"""
    overall_stats = {
        'total_tasks': len(all_results),
        'total_evaluations': 0,
        'overall_accuracy': 0,
        'task_breakdown': {}
    }

    total_accuracy_sum = 0
    total_evaluations = 0

    for task_name, results in all_results.items():
        task_stats = {
            'evaluations': 0,
            'avg_accuracy': 0,
            'status': 'completed'
        }

        # Handle different result formats
        if isinstance(results, dict):
            if 'task_metrics' in results:
                # Handle final_report.json format with task_metrics
                accuracies = []
                for test_case, metrics in results['task_metrics'].items():
                    if isinstance(metrics, dict) and 'accuracy' in metrics:
                        if isinstance(metrics['accuracy'], dict) and 'mean' in metrics['accuracy']:
                            accuracies.append(metrics['accuracy']['mean'])
                        elif isinstance(metrics['accuracy'], (int, float)):
                            accuracies.append(metrics['accuracy'])

                if accuracies:
                    task_stats['avg_accuracy'] = sum(accuracies) / len(accuracies)
                    task_stats['evaluations'] = len(accuracies)
                    total_accuracy_sum += sum(accuracies)
                    total_evaluations += len(accuracies)
            elif 'accuracy' in results:
                if isinstance(results['accuracy'], (int, float)):
                    task_stats['avg_accuracy'] = results['accuracy']
                    task_stats['evaluations'] = 1
                    total_accuracy_sum += results['accuracy']
                    total_evaluations += 1
            elif 'results' in results:
                # Handle nested results
                accuracies = []
                for result in results['results']:
                    if isinstance(result, dict) and 'accuracy' in result:
                        accuracies.append(result['accuracy'])

                if accuracies:
                    task_stats['avg_accuracy'] = sum(accuracies) / len(accuracies)
                    task_stats['evaluations'] = len(accuracies)
                    total_accuracy_sum += sum(accuracies)
                    total_evaluations += len(accuracies)

        overall_stats['task_breakdown'][task_name] = task_stats
        overall_stats['total_evaluations'] += task_stats['evaluations']

    if total_evaluations > 0:
        overall_stats['overall_accuracy'] = total_accuracy_sum / total_evaluations

    return overall_stats

def print_comprehensive_summary(model_id, api_provider, all_results, overall_stats):
    """Print comprehensive summary"""
    print("=" * 80)
    print("🏆 COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"🤖 Model: {model_id}")
    if api_provider:
        print(f"🌐 API Provider: {api_provider.upper()}")
    print(f"📊 Overall Performance: {overall_stats['overall_accuracy']:.1%}")
    print(f"📝 Total Tasks: {overall_stats['total_tasks']}")
    print(f"🧪 Total Evaluations: {overall_stats['total_evaluations']}")

    print("\n" + "-" * 80)
    print("📋 TASK PERFORMANCE BREAKDOWN")
    print("-" * 80)

    # If we have a main_report, show the detailed breakdown
    if 'main_report' in all_results and 'task_metrics' in all_results['main_report']:
        task_metrics = all_results['main_report']['task_metrics']

        # Group tasks by their base name (e.g., algebraic_sequence_6 -> algebraic_sequence)
        task_groups = {}
        for task_name, metrics in task_metrics.items():
            if '_avg' in task_name:
                continue  # Skip average entries for now

            # Extract base task name (e.g., algebraic_sequence from algebraic_sequence_6)
            base_task = '_'.join(task_name.split('_')[:-1]) if task_name.split('_')[-1].isdigit() else task_name

            if base_task not in task_groups:
                task_groups[base_task] = []

            task_groups[base_task].append({
                'name': task_name,
                'accuracy': metrics['accuracy']['mean'],
                'evaluations': 1
            })

        # Display task groups
        for base_task, subtasks in task_groups.items():
            avg_accuracy = sum(t['accuracy'] for t in subtasks) / len(subtasks)
            total_evaluations = sum(t['evaluations'] for t in subtasks)

            accuracy_str = f"{avg_accuracy:.1%}"
            status_emoji = "✅" if avg_accuracy >= 0.9 else "⚠️" if avg_accuracy >= 0.7 else "❌"

            print(f"{status_emoji} {base_task.upper().replace('_', ' ')}")
            print(f"    Overall Accuracy: {accuracy_str}")
            print(f"    Total Evaluations: {total_evaluations}")

            # Show breakdown by list size/test case
            for subtask in subtasks:
                size = subtask['name'].split('_')[-1] if subtask['name'].split('_')[-1].isdigit() else 'default'
                print(f"      • Size {size}: {subtask['accuracy']:.1%}")
            print()
    else:
        # Fallback: use the old method
        sorted_tasks = sorted(
            overall_stats['task_breakdown'].items(),
            key=lambda x: x[1]['avg_accuracy'],
            reverse=True
        )

        for task_name, stats in sorted_tasks:
            if stats['evaluations'] == 0:
                continue  # Skip tasks with no evaluations

            accuracy_str = f"{stats['avg_accuracy']:.1%}" if stats['avg_accuracy'] > 0 else "N/A"
            status_emoji = "✅" if stats['avg_accuracy'] >= 0.9 else "⚠️" if stats['avg_accuracy'] >= 0.7 else "❌"

            print(f"{status_emoji} {task_name.upper().replace('_', ' ')}")
            print(f"    Accuracy: {accuracy_str}")
            print(f"    Evaluations: {stats['evaluations']}")
            print()

    # Performance categories
    if 'main_report' in all_results and 'task_metrics' in all_results['main_report']:
        task_metrics = all_results['main_report']['task_metrics']
        task_groups = {}

        for task_name, metrics in task_metrics.items():
            if '_avg' in task_name:
                continue
            base_task = '_'.join(task_name.split('_')[:-1]) if task_name.split('_')[-1].isdigit() else task_name
            if base_task not in task_groups:
                task_groups[base_task] = []
            task_groups[base_task].append(metrics['accuracy']['mean'])

        # Calculate averages for each task group
        excellent_tasks = []
        good_tasks = []
        needs_work = []

        for base_task, accuracies in task_groups.items():
            avg_accuracy = sum(accuracies) / len(accuracies)
            if avg_accuracy >= 0.9:
                excellent_tasks.append(base_task)
            elif avg_accuracy >= 0.7:
                good_tasks.append(base_task)
            else:
                needs_work.append(base_task)
    else:
        # Fallback method
        excellent_tasks = [name for name, stats in overall_stats['task_breakdown'].items() if stats['avg_accuracy'] >= 0.9 and stats['evaluations'] > 0]
        good_tasks = [name for name, stats in overall_stats['task_breakdown'].items() if 0.7 <= stats['avg_accuracy'] < 0.9 and stats['evaluations'] > 0]
        needs_work = [name for name, stats in overall_stats['task_breakdown'].items() if stats['avg_accuracy'] < 0.7 and stats['evaluations'] > 0]

    print("-" * 80)
    print("🏅 PERFORMANCE CATEGORIES")
    print(f"🌟 Excellent (≥90%): {len(excellent_tasks)} tasks")
    for task in excellent_tasks:
        print(f"    • {task.replace('_', ' ').title()}")

    if good_tasks:
        print(f"🔄 Good (70-89%): {len(good_tasks)} tasks")
        for task in good_tasks:
            print(f"    • {task.replace('_', ' ').title()}")

    if needs_work:
        print(f"⚠️ Needs Improvement (<70%): {len(needs_work)} tasks")
        for task in needs_work:
            print(f"    • {task.replace('_', ' ').title()}")

    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive summary of evaluation results')
    parser.add_argument('model_tag', help='Model tag (e.g., gpt-5-nano)')
    parser.add_argument('--api_provider', help='API provider (openai or gemini)')
    parser.add_argument('--results_dir', help='Base results directory', default='Results')

    args = parser.parse_args()

    model_tag = args.model_tag
    results_base = args.results_dir

    # Load results
    results_dir = os.path.join(results_base, model_tag)

    print(f"📁 Loading results for {model_tag}...")
    print(f"   Results directory: {results_dir}")

    all_results = load_results_from_directory(results_dir)

    if not all_results:
        print(f"❌ No results found for {model_tag}")
        sys.exit(1)

    print(f"   Loaded {len(all_results)} task results")

    # Calculate overall metrics
    overall_stats = calculate_overall_metrics(all_results)

    # Print comprehensive summary
    print_comprehensive_summary(model_tag, args.api_provider, all_results, overall_stats)

    # Save detailed summary
    summary_file = os.path.join('logs', model_tag, 'comprehensive_summary.json')
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    summary_data = {
        'model_tag': model_tag,
        'api_provider': args.api_provider,
        'overall_stats': overall_stats,
        'detailed_results': all_results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n💾 Detailed summary saved to: {summary_file}")
    print(f"🎉 Evaluation analysis complete!")

if __name__ == "__main__":
    main()
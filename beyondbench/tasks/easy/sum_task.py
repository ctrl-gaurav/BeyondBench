import random
import logging
from tqdm import tqdm
import os
from ...core.base_task import BaseTask
from ...utils.parsing import parse_sum

class SumTask(BaseTask):
    """Implementation of the sum task"""
    
    @property
    def task_name(self):
        return "sum"
    
    def generate_data(self, list_size, include_negatives=True):
        """Generate random lists of numbers within specified range"""
        
        if self.seed is not None:
            random.seed(self.seed)
            
        if include_negatives:
            return [random.sample(range(self.min_val, self.max_val + 1), list_size) 
                    for _ in range(self.num_samples)]
        else:
            return [random.sample(range(1, self.max_val + 1), list_size) 
                    for _ in range(self.num_samples)]
    
    def create_prompt(self, data_point):
        """Create prompt for sum task"""
        return (f"Add the following list of numbers:\n{data_point}\n\n"
                f"Provide the sum. Your final answer must be in the format "
                f"\\boxed{{answer}} at the end.")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for sum task"""
        ground_truth = sum(data_point)
        parsed_answer = parse_sum(response)
        instruction_followed = parsed_answer is not None
        accuracy = 0
        
        # Convert the parsed answer to the appropriate type for comparison
        if parsed_answer is not None:
            if isinstance(parsed_answer, str):
                try:
                    parsed_answer = int(float(parsed_answer))
                except ValueError:
                    # Keep as string if conversion fails
                    pass
            
            # Check accuracy
            if isinstance(parsed_answer, (int, float)):
                accuracy = 1 if parsed_answer == ground_truth else 0
        
        return {
            "input_list": data_point,
            "sum": ground_truth,
            "calculated_answer": parsed_answer if isinstance(parsed_answer, (int, float)) else None,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self, list_sizes, include_negatives=True):
        """Run evaluation for multiple list sizes"""
        all_metrics = []
        
        for list_size in list_sizes:
            logging.info(f"\n{'='*40}\nEvaluating sum task with list size {list_size}\n{'='*40}")
            
            # Generate evaluation data
            data = self.generate_data(list_size, include_negatives)
            
            # Run each fold
            for fold in range(1, self.num_folds + 1):
                metrics = self.run_fold(data, list_size, fold)
                metrics['list_size'] = list_size
                metrics['include_negatives'] = include_negatives
                all_metrics.append(metrics)
        
        return all_metrics
import random
import logging
from tqdm import tqdm
import os
from ...utils.parsing import parse_count
from ...core.base_task import BaseTask

class EvenCountTask(BaseTask):
    """Implementation of the even count task"""
    
    @property
    def task_name(self):
        return "even_count"
    
    def generate_data(self, list_size, include_negatives=True):
        """Generate random lists of numbers for counting even numbers"""
        
        if self.seed is not None:
            random.seed(self.seed)
            
        data = []
        if include_negatives:
            data = [random.sample(range(-100, 101), list_size) 
                    for _ in range(self.num_samples)]
        else:
            data = [random.sample(range(1, 101), list_size) 
                    for _ in range(self.num_samples)]
        return data
    
    def create_prompt(self, data_point):
        """Create prompt for even count task"""
        return (f"Count the even numbers from the following list of numbers:\n{data_point}\n\n"
                f"Provide the final count of even numbers. Your final answer must be in the format "
                f"\\boxed{{answer}} at the end.")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for even count task"""
        ground_truth = len([num for num in data_point if num % 2 == 0])
        answer = parse_count(response)
        instruction_followed = answer is not None
        
        accuracy = 0
        if answer is not None and isinstance(answer, int):
            accuracy = 1 if answer == ground_truth else 0
        
        return {
            "input_list": data_point,
            "even_count": ground_truth,
            "calculated_answer": answer if isinstance(answer, int) else None,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self, list_sizes, include_negatives=True):
        """Run evaluation for multiple list sizes"""
        all_metrics = []
        
        for list_size in list_sizes:
            suffix = "mixed_numbers" if include_negatives else "positive_only"
            logging.info(f"\n{'='*40}\nEvaluating even count with list size {list_size} ({suffix})\n{'='*40}")
            
            # Generate evaluation data
            data = self.generate_data(list_size, include_negatives)
            
            # Run each fold
            for fold in range(1, self.num_folds + 1):
                metrics = self.run_fold(data, list_size, fold)
                metrics['list_size'] = list_size
                metrics['include_negatives'] = include_negatives
                all_metrics.append(metrics)
        
        return all_metrics
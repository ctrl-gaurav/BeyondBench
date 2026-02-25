import random
import logging
from tqdm import tqdm
from ...core.base_task import BaseTask
from ...utils.parsing import parse_arithmetic_result

class FindMaximumTask(BaseTask):
    """Implementation of the find maximum task"""
    
    @property
    def task_name(self):
        return "find_maximum"
    
    def generate_data(self, list_size):
        """Generate random lists of numbers for finding maximum"""
        
        if self.seed is not None:
            random.seed(self.seed)
            
        return [random.sample(range(self.min_val, self.max_val + 1), list_size) 
                for _ in range(self.num_samples)]
    
    def create_prompt(self, data_point):
        """Create prompt for find maximum task"""
        return (f"Find the maximum number from the given list of numbers. List = {data_point}.\n\n"
                f"Your final answer must be in the format \\boxed{{maximum}} at the end of your response.")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for find maximum task"""
        ground_truth = max(data_point)
        parsed_answer = parse_arithmetic_result(response)
        instruction_followed = parsed_answer is not None
        accuracy = 0
        
        try:
            # If parsed answer is correct, set accuracy to 1 regardless of instruction following
            accuracy = 1 if parsed_answer == ground_truth else 0
        except Exception as e:
            logging.debug(f"Comparison error: {e}")
            accuracy = 0
            
        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "parsed_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self, list_sizes):
        """Run evaluation for multiple list sizes"""
        all_metrics = []
        
        for list_size in list_sizes:
            logging.info(f"\n{'='*40}\nEvaluating finding maximum with list size {list_size}\n{'='*40}")
            
            # Generate evaluation data
            data = self.generate_data(list_size)
            
            # Run each fold
            for fold in range(1, self.num_folds + 1):
                metrics = self.run_fold(data, list_size, fold)
                metrics['list_size'] = list_size
                all_metrics.append(metrics)
        
        return all_metrics
import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_sorted_list

class SortingTask(BaseTask):
    """Implementation of the sorting task"""
    
    @property
    def task_name(self):
        return "sorting"
    
    def generate_data(self, list_size):
        """Generate random lists of numbers within specified range"""
        
        if self.seed is not None:
            random.seed(self.seed)
            
        return [random.sample(range(self.min_val, self.max_val + 1), list_size) 
                for _ in range(self.num_samples)]
    
    def create_prompt(self, data_point):
        """Create prompt for sorting task"""
        return (f"Sort the following list of numbers in ascending order:\n{data_point}\n\n"
                f"Provide the sorted list. Your final answer must be in the format "
                f"\\boxed{{sorted list}} at the end.")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for sorting task"""
        ground_truth = sorted(data_point)
        boxed_answer = parse_sorted_list(response)
        instruction_followed = boxed_answer is not None
        accuracy = 0
        
        if instruction_followed:
            try:
                # Allow for different order formats
                accuracy = 1 if boxed_answer == ground_truth else 0
            except Exception as e:
                logging.debug(f"Comparison error: {e}")
                accuracy = 0
        
        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "parsed_answer": boxed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self, list_sizes):
        """Run evaluation for multiple list sizes"""
        all_metrics = []
        
        for list_size in list_sizes:
            logging.info(f"\n{'='*40}\nEvaluating sorting with list size {list_size}\n{'='*40}")
            
            # Generate evaluation data
            data = self.generate_data(list_size)
            
            # Run each fold
            for fold in range(1, self.num_folds + 1):
                metrics = self.run_fold(data, list_size, fold)
                metrics['list_size'] = list_size
                all_metrics.append(metrics)
        
        return all_metrics
import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count

class OddCountTask(BaseTask):
    """Implementation of the odd count task"""
    
    @property
    def task_name(self):
        return "odd_count"
    
    def generate_data(self, list_size):
        """Generate random lists of numbers for odd count evaluation"""
        
        if self.seed is not None:
            random.seed(self.seed)
            
        # You can adapt this from your generate_numbers_list function
        return [random.sample(range(self.min_val, self.max_val + 1), list_size) 
                for _ in range(self.num_samples)]
    
    def create_prompt(self, data_point):
        """Create prompt for odd count task"""
        return (f"Count the odd numbers from the following list of numbers:\n{data_point}\n\n"
                f"Provide the final count of odd numbers. Your final answer must be in the format "
                f"\\boxed{{answer}} at the end.")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for odd count task"""
        # Calculate ground truth: count of odd numbers
        ground_truth = len([num for num in data_point if num % 2 != 0])
        
        # Use your parsing functions
        answer = parse_count(response)
        instruction_followed = answer is not None
        
        # Evaluate accuracy
        accuracy = 0
        if isinstance(answer, (int, float)):
            accuracy = 1 if answer == ground_truth else 0
        
        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "parsed_answer": answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self, list_sizes):
        """Run evaluation for multiple list sizes"""
        all_metrics = []
        
        for list_size in list_sizes:
            logging.info(f"\n{'='*40}\nEvaluating odd count with list size {list_size}\n{'='*40}")
            
            # Generate evaluation data
            data = self.generate_data(list_size)
            
            # Run each fold
            for fold in range(1, self.num_folds + 1):
                metrics = self.run_fold(data, list_size, fold)
                metrics['list_size'] = list_size
                all_metrics.append(metrics)
        
        return all_metrics
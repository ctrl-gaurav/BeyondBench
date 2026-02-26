import random
import logging
from ...utils.parsing import parse_arithmetic_result
from ...core.base_task import BaseTask

class MultiplicationTask(BaseTask):
    """Implementation of the multiplication task"""
    
    @property
    def task_name(self):
        return "multiplication"
    
    def generate_data(self, list_size, include_negatives=True):
        """Generate random lists of numbers for multiplication"""
        
        if self.seed is not None:
            random.seed(self.seed)
            
        data = []
        for _ in range(self.num_samples):
            if include_negatives:
                numbers = [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
            else:
                numbers = [random.randint(1, self.max_val) for _ in range(list_size)]
            data.append(numbers)
        return data
    
    def create_prompt(self, data_point):
        """Create prompt for multiplication task"""
        return (f"Multiply the following list of numbers:\n{data_point}\n\n"
                f"Provide the product. Your final answer must be in the format "
                f"\\boxed{{answer}} at the end.")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for multiplication task"""
        ground_truth = self.get_product(data_point)
        
        # Parse the response using the utility function
        parsed_answer = parse_arithmetic_result(response)
        instruction_followed = parsed_answer is not None
        
        accuracy = 0
        if parsed_answer is not None:
            try:
                parsed_answer = int(float(parsed_answer))
            except (ValueError, TypeError):
                pass
            
        if (parsed_answer == ground_truth):
            accuracy = 1
        
        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "parsed_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def get_product(self, numbers):
        """Calculate the product of a list of numbers."""
        product = 1
        for number in numbers:
            product *= number
        return product
    
    def run_evaluation(self, list_sizes, include_negatives=True):
        """Run evaluation for multiple list sizes"""
        all_metrics = []
        
        for list_size in list_sizes:
            logging.info(f"\n{'='*40}\nEvaluating multiplication with list size {list_size}\n{'='*40}")
            
            # Generate evaluation data
            data = self.generate_data(list_size, include_negatives)
            
            # Run each fold
            for fold in range(1, self.num_folds + 1):
                metrics = self.run_fold(data, list_size, fold)
                metrics['list_size'] = list_size
                metrics['include_negatives'] = include_negatives
                all_metrics.append(metrics)
        
        return all_metrics
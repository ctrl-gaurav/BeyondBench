import random
import logging
from tqdm import tqdm
from ...core.base_task import BaseTask
from ...utils.parsing import parse_arithmetic_result

class DivisionTask(BaseTask):
    """Implementation of the division task"""
    
    @property
    def task_name(self):
        return "division"
    
    def generate_data(self, **kwargs):
        """Generate random pairs of numbers for division"""
        
        if self.seed is not None:
            random.seed(self.seed)
            
        data = []
        
        for _ in range(self.num_samples):
            # Generate two numbers, ensuring no division by zero
            numerator = random.randint(self.min_val, self.max_val)
            # Ensure denominator is not zero
            if self.min_val <= 0 <= self.max_val:
                # Range includes zero — pick from non-zero values
                if self.min_val == 0 and self.max_val == 0:
                    denominator = 1  # Degenerate range, use fallback
                else:
                    denominator = 0
                    for _ in range(1000):
                        denominator = random.randint(self.min_val, self.max_val)
                        if denominator != 0:
                            break
                    if denominator == 0:
                        denominator = 1  # Fallback
            else:
                denominator = random.randint(self.min_val, self.max_val)
                
            data.append([numerator, denominator])
            
        return data
    
    def create_prompt(self, data_point):
        """Create prompt for division task"""
        return (f"Divide {data_point[0]} by {data_point[1]}.\n\n"
                f"Provide the answer as a floating point number. "
                f"Your final answer must be in the format \\boxed{{answer}} at the end.")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for division task"""
        ground_truth = data_point[0] / data_point[1]
        
        parsed_answer = parse_arithmetic_result(response)
        instruction_followed = parsed_answer is not None
        
        accuracy = 0
        if parsed_answer is not None and ground_truth is not None:
            # Compare with 2 decimal precision
            if round(parsed_answer, 2) == round(ground_truth, 2):
                accuracy = 1
        
        return {
            "numerator": data_point[0],
            "denominator": data_point[1],
            "ground_truth": ground_truth,
            "parsed_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self):
        """Run evaluation for division task"""
        all_metrics = []
        
        logging.info(f"\n{'='*40}\nEvaluating division task\n{'='*40}")
        
        # Generate evaluation data
        data = self.generate_data()
        
        # Run each fold
        for fold in range(1, self.num_folds + 1):
            metrics = self.run_fold(data, "division", fold)
            all_metrics.append(metrics)
        
        return all_metrics
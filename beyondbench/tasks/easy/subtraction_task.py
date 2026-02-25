import random
import logging
from tqdm import tqdm
from ...core.base_task import BaseTask
from ...utils.parsing import parse_arithmetic_result

class SubtractionTask(BaseTask):
    """Implementation of the subtraction task"""
    
    @property
    def task_name(self):
        return "subtraction"
    
    def generate_data(self):
        """Generate pairs of numbers for subtraction"""
        
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        
        for _ in range(self.num_samples):
            if self.min_val < 0:  # Include negative numbers if range allows
                a = random.randint(self.min_val, self.max_val)
                b = random.randint(self.min_val, self.max_val)
            else:
                a = random.randint(self.min_val, self.max_val)
                b = random.randint(self.min_val, self.max_val)
            
            data.append([a, b])
        
        return data
    
    def create_prompt(self, data_point):
        """Create prompt for subtraction task"""
        return (f"Can you subtract {data_point[0]} from {data_point[1]} and provide your final answer "
                f"in \\boxed{{answer}} format at the end of your response.")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for subtraction task"""
        ground_truth = data_point[1] - data_point[0]
        parsed_answer = parse_arithmetic_result(response)
        instruction_followed = parsed_answer is not None
        accuracy = 0
        
        if (ground_truth == parsed_answer):
            accuracy = 1
        
        return {
            "numbers": data_point,
            "ground_truth": ground_truth,
            "parsed_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self):
        """Run evaluation for subtraction task"""
        all_metrics = []
        
        logging.info(f"\n{'='*40}\nEvaluating subtraction task\n{'='*40}")
        
        # Generate evaluation data
        data = self.generate_data()
        
        # Run each fold
        for fold in range(1, self.num_folds + 1):
            metrics = self.run_fold(data, "subtraction", fold)
            all_metrics.append(metrics)
        
        return all_metrics
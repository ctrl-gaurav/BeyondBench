import random
import logging
from tqdm import tqdm
from ...core.base_task import BaseTask
from ...utils.parsing import parse_arithmetic_result

class MedianTask(BaseTask):
    @property
    def task_name(self):
        return "median"
    
    def generate_data(self, list_size):
        
        if self.seed is not None:
            random.seed(self.seed)
            
        data = []
        for _ in range(self.num_samples):
            numbers = [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
            # Calculate ground truth
            sorted_nums = sorted(numbers)
            if len(sorted_nums) % 2 == 0:
                # Even number of elements
                median_value = (sorted_nums[len(sorted_nums)//2 - 1] + sorted_nums[len(sorted_nums)//2]) / 2
            else:
                # Odd number of elements
                median_value = sorted_nums[len(sorted_nums)//2]
            data.append({"input_list": numbers, "median": median_value})
        return data
    
    def create_prompt(self, data_point):
        return (f"Find the median value of the following list of numbers:\n{data_point['input_list']}\n\n"
                f"The median is the middle value when the list is sorted. If there is an even number of elements, "
                f"the median is the average of the two middle values. Your final answer must be in the format "
                f"\\boxed{{median value}} at the end.")
    
    def evaluate_response(self, response, data_point):
        parsed_answer = parse_arithmetic_result(response)
        instruction_followed = parsed_answer is not None
        accuracy = 0
        
        if instruction_followed:
            # Allow for small floating point differences
            accuracy = 1 if abs(parsed_answer - data_point['median']) < 1e-6 else 0
        
        return {
            "input_list": data_point['input_list'],
            "ground_truth": data_point['median'],
            "parsed_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self, list_sizes):
        all_metrics = []
        
        for list_size in list_sizes:
            logging.info(f"\n{'='*40}\nEvaluating median calculation with list size {list_size}\n{'='*40}")
            
            # Generate evaluation data
            data = self.generate_data(list_size)
            
            # Run each fold
            for fold in range(1, self.num_folds + 1):
                metrics = self.run_fold(data, list_size, fold)
                metrics['list_size'] = list_size
                all_metrics.append(metrics)
        
        return all_metrics
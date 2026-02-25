import random
import logging
from collections import Counter
from tqdm import tqdm
from ...core.base_task import BaseTask
from ...utils.parsing import parse_arithmetic_result

class ModeTask(BaseTask):
    @property
    def task_name(self):
        return "mode"
    
    def generate_data(self, list_size):
        
        if self.seed is not None:
            random.seed(self.seed)
            
        data = []
        for _ in range(self.num_samples):
            # Start with random numbers
            base_numbers = [random.randint(self.min_val, self.max_val) for _ in range(list_size - 2)]
            
            # Generate a repeating element
            mode_value = random.randint(self.min_val, self.max_val)
            
            # Decide how many times to repeat (at least 2, but randomized)
            repeat_count = random.randint(2, min(4, list_size))
            
            # Ensure we don't exceed list_size
            repeat_count = min(repeat_count, list_size - len(base_numbers))
            
            # Add the repeating element
            numbers = base_numbers + [mode_value] * repeat_count
            
            # Shuffle the list
            random.shuffle(numbers)
            
            # Calculate mode(s) - there could be multiple modes
            counter = Counter(numbers)
            max_count = max(counter.values())
            modes = [num for num, count in counter.items() if count == max_count]
            
            # Sort modes for consistency
            modes.sort()
            
            data.append({"input_list": numbers, "modes": modes})
        return data
    
    def create_prompt(self, data_point):
        return (f"Find the mode(s) of the following list of numbers:\n{data_point['input_list']}\n\n"
                f"The mode is the value that appears most frequently. If multiple values appear with the same "
                f"highest frequency, return all of them. Your final answer must be in the format "
                f"\\boxed{{mode(s)}} at the end. If there are multiple modes, list them separated by commas.")
    
    def evaluate_response(self, response, data_point):
        parsed_answer = parse_arithmetic_result(response)
        instruction_followed = parsed_answer is not None
        accuracy = 0
        
        if instruction_followed:
            # Both lists should contain the same elements, order doesn't matter
            sorted_parsed = sorted(parsed_answer)
            sorted_ground_truth = sorted(data_point['modes'])
            accuracy = 1 if sorted_parsed == sorted_ground_truth else 0
        
        return {
            "input_list": data_point['input_list'],
            "ground_truth": data_point['modes'],
            "parsed_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self, list_sizes):
        all_metrics = []
        
        for list_size in list_sizes:
            logging.info(f"\n{'='*40}\nEvaluating mode calculation with list size {list_size}\n{'='*40}")
            
            # Generate evaluation data
            data = self.generate_data(list_size)
            
            # Run each fold
            for fold in range(1, self.num_folds + 1):
                metrics = self.run_fold(data, list_size, fold)
                metrics['list_size'] = list_size
                all_metrics.append(metrics)
        
        return all_metrics
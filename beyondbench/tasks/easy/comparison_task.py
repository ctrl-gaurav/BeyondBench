import random
import logging
import re
from tqdm import tqdm
from ...core.base_task import BaseTask

class ComparisonTask(BaseTask):
    """Implementation of the number comparison task"""
    
    @property
    def task_name(self):
        return "comparison"
    
    def generate_data(self):
        """Generate pairs of numbers with equal distribution of comparison operators"""
        
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        
        # Calculate number of samples for each comparison type
        samples_per_type = self.num_samples // 3
        remaining = self.num_samples % 3
        
        # Generate "greater than" samples
        for _ in range(samples_per_type + (1 if remaining > 0 else 0)):
            a = random.randint(self.min_val, self.max_val)
            b = random.randint(self.min_val, a - 1) if a > self.min_val else max(self.min_val, a - 1)
            data.append({"num1": a, "num2": b, "relation": ">", "answer": "greater than"})

        # Generate "less than" samples
        for _ in range(samples_per_type + (1 if remaining > 1 else 0)):
            a = random.randint(self.min_val, self.max_val)
            b = random.randint(a + 1, self.max_val) if a < self.max_val else min(self.max_val, a + 1)
            data.append({"num1": a, "num2": b, "relation": "<", "answer": "less than"})
        
        # Generate "equal to" samples
        for _ in range(samples_per_type):
            a = random.randint(self.min_val, self.max_val)
            data.append({"num1": a, "num2": a, "relation": "=", "answer": "equal to"})
        
        # Shuffle data
        random.shuffle(data)
        return data
    
    def create_prompt(self, data_point):
        """Create prompt for number comparison task"""
        return (f"Compare the following two numbers and determine their relationship:\n\n"
                f"Number 1: {data_point['num1']}\n"
                f"Number 2: {data_point['num2']}\n\n"
                f"Is Number 1 greater than, less than, or equal to Number 2? "
                f"Your final answer must be in the format \\boxed{{relation}} at the end, "
                f"where 'relation' is one of: 'greater than', 'less than', or 'equal to'.")
    
    def parse_comparison_answer(self, response):
        """Extract comparison answer from boxed response"""
        clean_response = response.replace('\n', ' ')

        # Try multiple boxed patterns - handle nested braces like \text{...}
        boxed_patterns = [
            # Standard boxed with nested braces (e.g., \boxed{\text{greater than}})
            r'\\boxed\{\\text\{([^}]+)\}\}',
            r'\\boxed\{\\textbf\{([^}]+)\}\}',
            r'\\boxed\{\\mathrm\{([^}]+)\}\}',
            # Standard boxed without nested braces
            r'\\boxed\{([^{}]+)\}',
            # Boxed with any nested content - greedy match between outer braces
            r'\\boxed\{(.+?)\}(?:\s|$|\.|,)',
        ]

        answer = None
        for pattern in boxed_patterns:
            match = re.search(pattern, clean_response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean LaTeX commands from the extracted answer
                answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
                answer = re.sub(r'\\textbf\{([^}]*)\}', r'\1', answer)
                answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)
                answer = re.sub(r'\\[a-zA-Z]+', '', answer).strip()
                break

        if not answer:
            # Fallback: look for comparison keywords in the last sentence
            last_sentences = clean_response.split('.')[-3:]
            text = ' '.join(last_sentences).lower()
            if 'greater than' in text:
                return 'greater than'
            elif 'less than' in text:
                return 'less than'
            elif 'equal to' in text:
                return 'equal to'
            return None

        answer = answer.lower()

        # Normalize variations of answers
        if any(term in answer for term in ['greater', '>', 'more', 'larger', 'bigger']):
            return 'greater than'
        elif any(term in answer for term in ['less', '<', 'smaller', 'lower']):
            return 'less than'
        elif any(term in answer for term in ['equal', '=', 'same']):
            return 'equal to'

        return None
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for comparison task"""
        parsed_answer = self.parse_comparison_answer(response)
        instruction_followed = parsed_answer is not None
        accuracy = 0
        
        if instruction_followed:
            accuracy = 1 if parsed_answer == data_point['answer'] else 0
        
        return {
            "num1": data_point['num1'],
            "num2": data_point['num2'],
            "expected_relation": data_point['answer'],
            "parsed_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self):
        """Run evaluation for number comparison task"""
        all_metrics = []
        
        logging.info(f"\n{'='*40}\nEvaluating number comparison task\n{'='*40}")
        
        # Generate evaluation data
        data = self.generate_data()
        
        # Run each fold
        for fold in range(1, self.num_folds + 1):
            metrics = self.run_fold(data, "comparison", fold)
            all_metrics.append(metrics)
        
        return all_metrics
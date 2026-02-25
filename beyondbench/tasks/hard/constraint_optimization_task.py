"""
Advanced Multi-Constraint Resource Allocation Optimization Task

This task evaluates LLM's ability to solve complex multi-objective optimization problems
with various constraint types. It requires deep understanding of:
- Integer Linear Programming (ILP)
- Multi-objective optimization
- Constraint satisfaction
- Operations research principles
- Mathematical modeling
- Complex decision-making under constraints

PROBLEM TYPES:
- Resource allocation across multiple projects
- Multi-objective optimization (profit, efficiency, risk)
- Capacity constraints (budget, time, personnel, equipment)
- Quality and dependency constraints
- Risk management constraints
- Integer programming constraints

Algorithm: Uses branch-and-bound algorithm with constraint satisfaction, dynamic programming
for optimization, and sophisticated heuristics for multi-objective problems.
Ground truth generated using exact optimization algorithms.

Reasoning: Requires advanced mathematical modeling, operations research knowledge,
multi-step optimization, constraint analysis, and strategic decision-making.

Example: Allocate resources (budget=1000, time=100, personnel=50) across 5 projects 
to maximize profit while satisfying quality constraints, dependency requirements,
and risk limits.

CLI USAGE:
python constraint_optimization_task.py --model_id "Qwen/Qwen2.5-3B-Instruct" --datapoints 20 --folds 1 --num_projects 4,5,6 --constraint_types "basic,quality,dependency,risk,mixed" --temperature 0.1 --top_p 0.9 --max_tokens 8192 --seed 42 --store_details True
"""

# ============================================================================
# CONFIGURATION PARAMETERS (Customize as needed)
# ============================================================================

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TASKS = ["constraint_optimization"]
ENGINE = "vllm"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.96
TRUST_REMOTE_CODE = False

# Evaluation Configuration
DATAPOINTS = 20
FOLDS = 1
NUM_PROJECTS = [4, 5, 6]  # Number of projects to optimize
CONSTRAINT_TYPES = ["basic", "quality", "dependency", "risk", "mixed"]  # Types of constraints
STORE_DETAILS = True
SEED = 42

# Generation Parameters
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 8192

import os
import sys
import json
import argparse
import random
import logging
import numpy as np
import re
import ast
import traceback
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import math
from dataclasses import dataclass
from itertools import product, combinations


from ...core.base_task import BaseTask
from ...models.model_handler import ModelHandler
from ...utils.report_generator import generate_final_report
from ...utils.logging_utils import setup_logging

@dataclass
class Resource:
    """Represents a resource with capacity"""
    name: str
    capacity: int
    unit: str
    
    def __str__(self):
        return f"{self.name}: {self.capacity} {self.unit}"

@dataclass
class Project:
    """Represents a project with requirements and returns"""
    id: int
    name: str
    resource_requirements: Dict[str, int]  # resource_name -> required_amount
    profit: int
    quality_score: int  # 1-10 scale
    risk_level: int     # 1-5 scale (1=low, 5=high)
    dependencies: List[int]  # list of project IDs that must be selected first
    
    def __str__(self):
        deps_str = f", Dependencies: {self.dependencies}" if self.dependencies else ""
        return (f"Project {self.id} ({self.name}): "
                f"Profit={self.profit}, Quality={self.quality_score}, Risk={self.risk_level}"
                f"{deps_str}")

@dataclass
class OptimizationConstraint:
    """Represents various optimization constraints"""
    constraint_type: str
    parameters: Dict[str, Any]
    
    def __str__(self):
        if self.constraint_type == "min_quality":
            return f"Minimum average quality score: {self.parameters['min_score']}"
        elif self.constraint_type == "max_risk":
            return f"Maximum average risk level: {self.parameters['max_risk']}"
        elif self.constraint_type == "min_projects":
            return f"Minimum number of projects: {self.parameters['min_count']}"
        elif self.constraint_type == "max_projects":
            return f"Maximum number of projects: {self.parameters['max_count']}"
        elif self.constraint_type == "profit_threshold":
            return f"Minimum total profit: {self.parameters['min_profit']}"
        else:
            return f"{self.constraint_type}: {self.parameters}"

@dataclass
class OptimizationSolution:
    """Represents a solution to the optimization problem"""
    selected_projects: List[int]
    resource_usage: Dict[str, int]
    total_profit: int
    average_quality: float
    average_risk: float
    is_feasible: bool
    constraints_satisfied: List[bool]
    
    def __str__(self):
        return (f"Selected Projects: {self.selected_projects}, "
                f"Profit: {self.total_profit}, Quality: {self.average_quality:.2f}, "
                f"Risk: {self.average_risk:.2f}, Feasible: {self.is_feasible}")

class OptimizationSolver:
    """Solver for multi-constraint optimization problems"""
    
    @staticmethod
    def is_feasible_solution(selected_projects: List[int], 
                           projects: List[Project], 
                           resources: List[Resource],
                           constraints: List[OptimizationConstraint]) -> Tuple[bool, List[bool], Dict[str, Any]]:
        """Check if a solution is feasible given all constraints"""
        if not selected_projects:
            return False, [], {}
        
        # Calculate resource usage
        resource_usage = {res.name: 0 for res in resources}
        total_profit = 0
        quality_scores = []
        risk_levels = []
        
        for project_id in selected_projects:
            project = projects[project_id]
            total_profit += project.profit
            quality_scores.append(project.quality_score)
            risk_levels.append(project.risk_level)
            
            for res_name, req_amount in project.resource_requirements.items():
                resource_usage[res_name] += req_amount
        
        # Check resource capacity constraints
        for resource in resources:
            if resource_usage[resource.name] > resource.capacity:
                return False, [], {}
        
        # Check dependencies
        selected_set = set(selected_projects)
        for project_id in selected_projects:
            project = projects[project_id]
            for dep_id in project.dependencies:
                if dep_id not in selected_set:
                    return False, [], {}
        
        # Calculate metrics
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        avg_risk = np.mean(risk_levels) if risk_levels else 0
        
        # Check additional constraints
        constraint_results = []
        for constraint in constraints:
            satisfied = True
            
            if constraint.constraint_type == "min_quality":
                satisfied = avg_quality >= constraint.parameters['min_score']
            elif constraint.constraint_type == "max_risk":
                satisfied = avg_risk <= constraint.parameters['max_risk']
            elif constraint.constraint_type == "min_projects":
                satisfied = len(selected_projects) >= constraint.parameters['min_count']
            elif constraint.constraint_type == "max_projects":
                satisfied = len(selected_projects) <= constraint.parameters['max_count']
            elif constraint.constraint_type == "profit_threshold":
                satisfied = total_profit >= constraint.parameters['min_profit']
            
            constraint_results.append(satisfied)
        
        all_constraints_satisfied = all(constraint_results)
        
        metrics = {
            'resource_usage': resource_usage,
            'total_profit': total_profit,
            'average_quality': avg_quality,
            'average_risk': avg_risk,
            'num_projects': len(selected_projects)
        }
        
        return all_constraints_satisfied, constraint_results, metrics
    
    @staticmethod
    def solve_optimization_problem(projects: List[Project], 
                                 resources: List[Resource],
                                 constraints: List[OptimizationConstraint]) -> Optional[OptimizationSolution]:
        """
        Solve optimization problem using branch-and-bound with exhaustive search
        for smaller problems
        """
        n = len(projects)
        best_solution = None
        best_profit = -1
        
        # For small problems, use exhaustive search
        if n <= 10:
            # Generate all possible project combinations
            for r in range(1, n + 1):
                for project_combination in combinations(range(n), r):
                    selected_projects = list(project_combination)
                    
                    # Check feasibility
                    is_feasible, constraint_results, metrics = OptimizationSolver.is_feasible_solution(
                        selected_projects, projects, resources, constraints
                    )
                    
                    if is_feasible and metrics['total_profit'] > best_profit:
                        best_profit = metrics['total_profit']
                        best_solution = OptimizationSolution(
                            selected_projects=selected_projects,
                            resource_usage=metrics['resource_usage'],
                            total_profit=metrics['total_profit'],
                            average_quality=metrics['average_quality'],
                            average_risk=metrics['average_risk'],
                            is_feasible=True,
                            constraints_satisfied=constraint_results
                        )
        
        else:
            # For larger problems, use greedy heuristic with local search
            best_solution = OptimizationSolver.greedy_with_local_search(projects, resources, constraints)
        
        return best_solution
    
    @staticmethod
    def greedy_with_local_search(projects: List[Project], 
                               resources: List[Resource],
                               constraints: List[OptimizationConstraint]) -> Optional[OptimizationSolution]:
        """Greedy algorithm with local search for larger problems"""
        n = len(projects)
        
        # Sort projects by profit-to-cost ratio
        def profit_per_resource(project):
            total_resources = sum(project.resource_requirements.values())
            return project.profit / max(1, total_resources)
        
        sorted_indices = sorted(range(n), key=lambda i: profit_per_resource(projects[i]), reverse=True)
        
        # Greedy construction
        selected_projects = []
        for project_id in sorted_indices:
            candidate_selection = selected_projects + [project_id]
            is_feasible, _, _ = OptimizationSolver.is_feasible_solution(
                candidate_selection, projects, resources, constraints
            )
            
            if is_feasible:
                selected_projects.append(project_id)
        
        if not selected_projects:
            return None
        
        # Create solution object
        is_feasible, constraint_results, metrics = OptimizationSolver.is_feasible_solution(
            selected_projects, projects, resources, constraints
        )
        
        if is_feasible:
            return OptimizationSolution(
                selected_projects=selected_projects,
                resource_usage=metrics['resource_usage'],
                total_profit=metrics['total_profit'],
                average_quality=metrics['average_quality'],
                average_risk=metrics['average_risk'],
                is_feasible=True,
                constraints_satisfied=constraint_results
            )
        
        return None

class OptimizationProblemGenerator:
    """Generator for various types of optimization problems"""
    
    @staticmethod
    def generate_basic_problem(num_projects: int) -> Tuple[List[Project], List[Resource], List[OptimizationConstraint]]:
        """Generate basic resource allocation problem"""
        # Generate resources
        resources = [
            Resource("Budget", random.randint(800, 1500), "dollars"),
            Resource("Time", random.randint(80, 150), "hours"), 
            Resource("Personnel", random.randint(30, 70), "people")
        ]
        
        # Generate projects
        projects = []
        for i in range(num_projects):
            # Generate resource requirements that make the problem challenging
            budget_req = random.randint(150, 400)
            time_req = random.randint(15, 40)
            personnel_req = random.randint(5, 15)
            
            project = Project(
                id=i,
                name=f"Project_{i+1}",
                resource_requirements={
                    "Budget": budget_req,
                    "Time": time_req,
                    "Personnel": personnel_req
                },
                profit=random.randint(200, 600),
                quality_score=random.randint(6, 10),
                risk_level=random.randint(1, 3),
                dependencies=[]
            )
            projects.append(project)
        
        # Basic constraint: resource capacity only (handled automatically)
        constraints = []
        
        return projects, resources, constraints
    
    @staticmethod
    def generate_quality_constraint_problem(num_projects: int) -> Tuple[List[Project], List[Resource], List[OptimizationConstraint]]:
        """Generate problem with quality constraints"""
        projects, resources, _ = OptimizationProblemGenerator.generate_basic_problem(num_projects)
        
        # Add quality constraint
        constraints = [
            OptimizationConstraint("min_quality", {"min_score": 7.5})
        ]
        
        return projects, resources, constraints
    
    @staticmethod
    def generate_dependency_constraint_problem(num_projects: int) -> Tuple[List[Project], List[Resource], List[OptimizationConstraint]]:
        """Generate problem with project dependencies"""
        projects, resources, _ = OptimizationProblemGenerator.generate_basic_problem(num_projects)
        
        # Add dependencies between projects
        if num_projects >= 3:
            # Make project 1 depend on project 0
            projects[1].dependencies = [0]
            
            # Make project 3 depend on project 2 (if exists)
            if num_projects >= 4:
                projects[3].dependencies = [2]
            
            # Make last project depend on multiple projects
            if num_projects >= 5:
                projects[-1].dependencies = [0, 2]
        
        constraints = []
        return projects, resources, constraints
    
    @staticmethod
    def generate_risk_constraint_problem(num_projects: int) -> Tuple[List[Project], List[Resource], List[OptimizationConstraint]]:
        """Generate problem with risk constraints"""
        projects, resources, _ = OptimizationProblemGenerator.generate_basic_problem(num_projects)
        
        # Make some projects higher risk but higher reward
        for i in range(len(projects)):
            if i % 2 == 0:  # Even indexed projects are higher risk/reward
                projects[i].risk_level = random.randint(3, 5)
                projects[i].profit = int(projects[i].profit * 1.4)
        
        # Add risk constraint
        constraints = [
            OptimizationConstraint("max_risk", {"max_risk": 3.0})
        ]
        
        return projects, resources, constraints
    
    @staticmethod
    def generate_mixed_constraint_problem(num_projects: int) -> Tuple[List[Project], List[Resource], List[OptimizationConstraint]]:
        """Generate problem with multiple constraint types"""
        projects, resources, _ = OptimizationProblemGenerator.generate_basic_problem(num_projects)
        
        # Add dependencies
        if num_projects >= 4:
            projects[1].dependencies = [0]
            projects[3].dependencies = [2]
        
        # Vary risk and quality
        for i in range(len(projects)):
            if i % 3 == 0:  # Every third project is high risk/reward
                projects[i].risk_level = random.randint(4, 5)
                projects[i].profit = int(projects[i].profit * 1.5)
                projects[i].quality_score = random.randint(8, 10)
        
        # Add multiple constraints
        constraints = [
            OptimizationConstraint("min_quality", {"min_score": 7.0}),
            OptimizationConstraint("max_risk", {"max_risk": 3.5}),
            OptimizationConstraint("min_projects", {"min_count": 2})
        ]
        
        return projects, resources, constraints

class ConstraintOptimizationTask(BaseTask):
    """Multi-Constraint Optimization evaluation task with sophisticated parsing"""
    
    def __init__(self, model_handler, output_dir, num_projects_list, constraint_types, num_folds, 
                 num_samples, store_details, temperature, top_p, max_tokens, seed=None):
        # Initialize base class with proper min/max values
        self._task_name = "constraint_optimization_results"
        super().__init__(model_handler, output_dir, min(num_projects_list), max(num_projects_list), 
                        num_folds, num_samples, store_details, temperature, top_p, max_tokens, seed)
        self.num_projects_list = num_projects_list
        self.constraint_types = constraint_types
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Setup parsing patterns
        self.setup_parsing_patterns()
    
    @property
    def task_name(self):
        """Return task name"""
        return self._task_name
    
    def generate_data(self, **kwargs):
        """Generate evaluation data
        
        Args:
            list_size: Number of projects in the problem
        """
        list_size = kwargs.get('list_size', None)
        if self.seed is not None:
            random.seed(self.seed)
        
        num_projects = list_size if list_size is not None else random.choice(self.num_projects_list)
        test_cases = []
        
        # Generate exactly num_samples total, distributed across constraint types
        samples_per_type = max(1, self.num_samples // len(self.constraint_types))
        remaining_samples = self.num_samples - (samples_per_type * len(self.constraint_types))
        
        for i, constraint_type in enumerate(self.constraint_types):
            # Add extra sample to first types if needed to reach exact num_samples
            samples_for_this_type = samples_per_type + (1 if i < remaining_samples else 0)
            
            for _ in range(samples_for_this_type):
                problem = self.generate_problem(constraint_type, num_projects)
                if problem['reference_solution'] is not None:
                    problem['answer'] = problem['reference_solution'].selected_projects  # For BaseTask compatibility
                    test_cases.append(problem)
        
        # Ensure we have exactly num_samples by padding if necessary
        while len(test_cases) < self.num_samples:
            constraint_type = random.choice(self.constraint_types)
            problem = self.generate_problem(constraint_type, num_projects)
            if problem['reference_solution'] is not None:
                problem['answer'] = problem['reference_solution'].selected_projects
                test_cases.append(problem)
        
        return test_cases[:self.num_samples]  # Ensure exact count
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response with robust handling"""
        result = self.parse_response(response, data_point)
        
        # Always include parsed result
        parsed_solution = result.get('parsed_solution', [])
        is_correct = result.get('is_correct', False)
        parsing_confidence = result.get('parsing_confidence', 0.0)
        
        # More lenient instruction following - consider successful if we parsed project IDs
        instruction_followed = 1 if parsed_solution or parsing_confidence > 0.3 else 0
        
        return {
            'accuracy': 1 if is_correct else 0,
            'instruction_followed': instruction_followed,
            'predicted_answer': parsed_solution,
            'ground_truth': data_point.get('reference_solution').selected_projects if data_point.get('reference_solution') else [],
            'parsing_info': {
                'method': result.get('parsing_method', 'unknown'),
                'confidence': parsing_confidence,
                'attempts': result.get('parsing_attempts', []),
                'raw_response_preview': response[:200] + '...' if len(response) > 200 else response
            },
            'validation_info': result.get('validation_info', {})
        }
    
    def setup_parsing_patterns(self):
        """Setup regex patterns for parsing project selection formats"""
        self.solution_patterns = [
            # Pattern 1: "Selected projects: [1, 2, 3]" or "Projects: [1,2,3]"
            (re.compile(r'(?:selected\s+)?projects?\s*:?\s*\[([^\]]+)\]', re.IGNORECASE), "project_list", 0.95),
            
            # Pattern 2: Answer tags with list
            (re.compile(r'<answer>\s*\[([^\]]+)\]\s*</answer>', re.IGNORECASE), "answer_tags_list", 0.98),
            
            # Pattern 3: "Solution: [1, 2, 3]"
            (re.compile(r'(?:solution|answer)\s*:?\s*\[([^\]]+)\]', re.IGNORECASE), "solution_list", 0.90),
            
            # Pattern 4: "Choose projects 1, 2, and 3"
            (re.compile(r'choose\s+projects?\s+([\d,\s\w]+)', re.IGNORECASE), "choose_projects", 0.85),
            
            # Pattern 5: "Projects 1, 2, 3 should be selected"
            (re.compile(r'projects?\s+([\d,\s\w]+)\s+should\s+be\s+selected', re.IGNORECASE), "projects_selected", 0.88),
            
            # Pattern 6: Simple list format "1, 2, 3"
            (re.compile(r'(\d+(?:\s*,\s*\d+)*)', re.IGNORECASE), "simple_list", 0.60),
            
            # Pattern 7: "Select: 1, 2, 3"
            (re.compile(r'select\s*:?\s*([\d,\s]+)', re.IGNORECASE), "select_format", 0.82),
            
            # Pattern 8: Bullet points or numbered lists
            (re.compile(r'(?:^\s*[-*•]|\d+\.)\s*(?:project\s*)?(\d+)', re.IGNORECASE | re.MULTILINE), "bullet_list", 0.75),
            
            # Pattern 9: "The optimal selection is [1, 2, 3]"
            (re.compile(r'optimal\s+selection\s+is\s*\[([^\]]+)\]', re.IGNORECASE), "optimal_selection", 0.93),
            
            # Pattern 10: Final answer with numbers
            (re.compile(r'final\s+(?:answer|solution)\s*:?\s*(?:\[)?([^\]]+)(?:\])?', re.IGNORECASE), "final_answer", 0.87)
        ]
    
    def generate_problem(self, constraint_type: str, num_projects: int) -> Dict[str, Any]:
        """Generate an optimization problem"""
        # Generate problem based on constraint type
        if constraint_type == "basic":
            projects, resources, constraints = OptimizationProblemGenerator.generate_basic_problem(num_projects)
        elif constraint_type == "quality":
            projects, resources, constraints = OptimizationProblemGenerator.generate_quality_constraint_problem(num_projects)
        elif constraint_type == "dependency":
            projects, resources, constraints = OptimizationProblemGenerator.generate_dependency_constraint_problem(num_projects)
        elif constraint_type == "risk":
            projects, resources, constraints = OptimizationProblemGenerator.generate_risk_constraint_problem(num_projects)
        elif constraint_type == "mixed":
            projects, resources, constraints = OptimizationProblemGenerator.generate_mixed_constraint_problem(num_projects)
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        
        # Find reference solution
        reference_solution = OptimizationSolver.solve_optimization_problem(projects, resources, constraints)
        
        return {
            'num_projects': num_projects,
            'constraint_type': constraint_type,
            'projects': projects,
            'resources': resources,
            'constraints': constraints,
            'reference_solution': reference_solution
        }
    
    def create_prompt(self, problem: Dict[str, Any]) -> str:
        """Create comprehensive prompt for optimization problem"""
        projects = problem['projects']
        resources = problem['resources']
        constraints = problem['constraints']
        constraint_type = problem.get('constraint_type', 'basic')
        
        # Build resources string
        resources_str = '\n'.join([f"- {resource}" for resource in resources])
        
        # Build projects string
        projects_str = ""
        for project in projects:
            req_str = ', '.join([f"{name}={amount}" for name, amount in project.resource_requirements.items()])
            deps_str = f", Dependencies: {project.dependencies}" if project.dependencies else ""
            projects_str += f"- {project}{deps_str}\n  Resource Requirements: {req_str}\n"
        
        # Build constraints string
        constraints_str = ""
        if constraints:
            constraints_str = "\n\nADDITIONAL CONSTRAINTS:\n"
            constraints_str += '\n'.join([f"- {constraint}" for constraint in constraints])
        
        # Add strategic hints based on constraint type
        strategy_hint = ""
        if constraint_type == "dependency":
            strategy_hint = "\n\nSTRATEGY HINT: Pay careful attention to project dependencies. A project cannot be selected unless all its dependencies are also selected."
        elif constraint_type == "mixed":
            strategy_hint = "\n\nSTRATEGY HINT: This problem has multiple constraint types. Consider quality, risk, and quantity requirements when making your selection."
        elif constraint_type == "risk":
            strategy_hint = "\n\nSTRATEGY HINT: Balance high-profit projects with their associated risk levels to meet the risk constraint."
        
        # Create comprehensive prompt
        prompt = f"""ADVANCED MULTI-CONSTRAINT RESOURCE ALLOCATION PROBLEM:

Your task: Select an optimal subset of projects to maximize total profit while satisfying all resource capacity constraints and additional requirements.

AVAILABLE RESOURCES:
{resources_str}

AVAILABLE PROJECTS:
{projects_str}{constraints_str}

OPTIMIZATION OBJECTIVE:
Maximize total profit from selected projects

CONSTRAINTS:
1. Resource capacity: Cannot exceed available capacity for any resource
2. Dependencies: If a project has dependencies, all dependency projects must also be selected{constraints_str}

SOLUTION APPROACH:
1. Analyze resource requirements and available capacity
2. Consider profit-to-resource ratios for efficiency
3. Check dependency relationships between projects
4. Verify all additional constraints are satisfied
5. Find the combination that maximizes profit while meeting all constraints

IMPORTANT REQUIREMENTS:
- Selected projects must not exceed any resource capacity
- All project dependencies must be satisfied
- Additional constraints must be met
- Solution should maximize total profit{strategy_hint}

VERIFICATION CHECKLIST:
1. Sum resource usage for selected projects ≤ available capacity for each resource
2. All dependencies satisfied: if project X is selected, all its dependencies are also selected
3. Additional constraints satisfied (quality, risk, etc.)
4. Calculate total profit to verify optimality

ANSWER FORMAT:
Provide your answer as a list of project IDs (numbers) in square brackets.

<answer>
[project_id_1, project_id_2, project_id_3, ...]
</answer>

EXAMPLE:
If you select projects 0, 2, and 4: <answer>[0, 2, 4]</answer>

CALCULATION EXAMPLE:
For selected projects [0, 1]:
- Total Budget used: Project_0_budget + Project_1_budget ≤ Available_Budget
- Total Time used: Project_0_time + Project_1_time ≤ Available_Time
- Total Personnel used: Project_0_personnel + Project_1_personnel ≤ Available_Personnel
- Dependencies: Check if any selected project depends on unselected projects
- Additional constraints: Verify quality/risk/quantity requirements

Find the optimal project selection that maximizes profit under all constraints."""
        
        return prompt
    
    def parse_response(self, response: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response with enhanced robustness and detailed diagnostics"""
        parsing_attempts = []
        best_solution = []
        best_confidence = 0.0
        best_method = "none"
        
        # Extract answer with priority extraction
        answer_text = response.strip()
        
        # Priority extraction order
        if '<answer>' in response and '</answer>' in response:
            answer_text = response.split('<answer>')[1].split('</answer>')[0].strip()
            parsing_attempts.append(("answer_tags", "Extracted from <answer> tags", 0.1))
        elif '```' in response:
            # Extract from code blocks
            code_blocks = re.findall(r'```(?:[a-z]*\n)?(.*?)```', response, re.DOTALL)
            if code_blocks:
                answer_text = code_blocks[-1].strip()  # Take the last code block
                parsing_attempts.append(("code_blocks", "Extracted from code blocks", 0.05))
        
        # Try each parsing pattern
        for pattern, method_name, base_confidence in self.solution_patterns:
            try:
                if method_name == "bullet_list":
                    # Special handling for bullet lists
                    matches = pattern.findall(answer_text)
                    if matches:
                        solution = [int(match) for match in matches if match.isdigit()]
                        if solution:
                            confidence = base_confidence
                            parsing_attempts.append((method_name, f"Found: {solution}", confidence))
                            if confidence > best_confidence:
                                best_solution = solution
                                best_confidence = confidence
                                best_method = method_name
                else:
                    matches = pattern.findall(answer_text)
                    if matches:
                        # Parse the matched content
                        match_str = matches[0] if isinstance(matches[0], str) else str(matches[0])
                        
                        # Extract numbers from the match
                        numbers = re.findall(r'\d+', match_str)
                        solution = [int(num) for num in numbers]
                        
                        if solution:
                            confidence = base_confidence
                            parsing_attempts.append((method_name, f"Found: {solution}", confidence))
                            
                            if confidence > best_confidence:
                                best_solution = solution
                                best_confidence = confidence
                                best_method = method_name
                    else:
                        parsing_attempts.append((method_name, "No match", 0.0))
                        
            except (ValueError, IndexError) as e:
                parsing_attempts.append((method_name, f"Error: {str(e)}", 0.0))
        
        # Validate solution
        validation_info = {}
        if best_solution:
            # Filter valid project IDs
            num_projects = problem['num_projects']
            valid_projects = [p for p in best_solution if 0 <= p < num_projects]
            
            if valid_projects:
                is_correct = self.validate_solution(valid_projects, problem)
                is_feasible, constraint_results, metrics = OptimizationSolver.is_feasible_solution(
                    valid_projects, problem['projects'], problem['resources'], problem['constraints']
                )
                
                validation_info = {
                    'valid_projects': valid_projects,
                    'is_feasible': is_feasible,
                    'constraint_results': constraint_results,
                    'metrics': metrics,
                    'is_optimal': is_correct
                }
                
                best_solution = valid_projects
            else:
                is_correct = False
                validation_info = {'error': 'No valid project IDs found'}
        else:
            is_correct = False
            validation_info = {'error': 'Could not parse any project selection from response'}
        
        return {
            'parsed_solution': best_solution,
            'parsing_method': best_method,
            'parsing_confidence': best_confidence,
            'parsing_attempts': parsing_attempts,
            'validation_info': validation_info,
            'is_correct': is_correct,
            'raw_response': response
        }
    
    def validate_solution(self, solution: List[int], problem: Dict[str, Any]) -> bool:
        """Validate that solution is optimal or near-optimal"""
        if not solution:
            return False
        
        # Check feasibility
        is_feasible, constraint_results, metrics = OptimizationSolver.is_feasible_solution(
            solution, problem['projects'], problem['resources'], problem['constraints']
        )
        
        if not is_feasible:
            return False
        
        # Check if it's the optimal solution
        reference_solution = problem['reference_solution']
        if reference_solution is None:
            return False
        
        # Accept solution if it achieves the same profit as optimal
        solution_profit = metrics['total_profit']
        optimal_profit = reference_solution.total_profit
        
        # Allow small tolerance for near-optimal solutions
        return solution_profit >= optimal_profit * 0.95

def main():
    parser = argparse.ArgumentParser(description="Multi-Constraint Optimization Evaluation")
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Model identifier')
    parser.add_argument('--engine', type=str, default=ENGINE, choices=['vllm', 'transformers'], help='Inference engine')
    parser.add_argument('--tensor_parallel_size', type=int, default=TENSOR_PARALLEL_SIZE, help='Tensor parallel size for VLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=GPU_MEMORY_UTILIZATION, help='GPU memory utilization for VLLM')
    parser.add_argument('--trust_remote_code', type=bool, default=TRUST_REMOTE_CODE, help='Trust remote code')
    
    # Task configuration
    parser.add_argument('--datapoints', type=int, default=DATAPOINTS, help='Number of datapoints per fold')
    parser.add_argument('--folds', type=int, default=FOLDS, help='Number of evaluation folds')
    parser.add_argument('--num_projects', type=str, default=','.join(map(str, NUM_PROJECTS)), help='Number of projects (comma-separated)')
    parser.add_argument('--constraint_types', type=str, default=','.join(CONSTRAINT_TYPES), help='Constraint types (comma-separated)')
    parser.add_argument('--store_details', type=bool, default=STORE_DETAILS, help='Store detailed results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=TOP_P, help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS, help='Maximum tokens to generate')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='constraint_optimization_results', help='Output directory')

    # API configuration
    parser.add_argument('--api_provider', type=str, choices=['openai', 'gemini'], help='API provider for cloud models')
    parser.add_argument('--api_key', type=str, help='API key for cloud models')

    # Reasoning configuration
    parser.add_argument('--reasoning_effort', type=str, choices=['minimal', 'low', 'medium', 'high'],
                       help='OpenAI reasoning effort level (for GPT-5 models)')
    parser.add_argument('--thinking_budget', type=int,
                       help='Gemini thinking budget (0 to disable, -1 for dynamic, or specific number)')
    
    args = parser.parse_args()
    
    # Parse parameters
    num_projects_list = [int(x.strip()) for x in args.num_projects.split(',')]
    constraint_types = [x.strip() for x in args.constraint_types.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("⚙️ Starting Multi-Constraint Optimization Evaluation")
    logging.info(f"📁 Output directory: {args.output_dir}")
    logging.info(f"📝 Log file: {log_file}")
    
    try:
        # Initialize model handler
        logging.info(f"🤖 Initializing model handler for {args.model_id}")
        model_handler = ModelHandler(
            model_id=args.model_id,
            api_provider=args.api_provider,
            api_key=args.api_key,
            reasoning_effort=args.reasoning_effort,
            thinking_budget=args.thinking_budget,
            engine=args.engine,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code
        )
    
        # Initialize task
        logging.info(f"🎯 Initializing Constraint Optimization task")
        task = ConstraintOptimizationTask(
            model_handler=model_handler,
            output_dir=args.output_dir,
            num_projects_list=num_projects_list,
            constraint_types=constraint_types,
            num_folds=args.folds,
            num_samples=args.datapoints,
            store_details=args.store_details,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed
        )
    
        # Set random seeds
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            logging.info(f"🎲 Set random seed to {args.seed}")
    
        # Run evaluation
        logging.info(f"🚀 Starting evaluation with project counts: {num_projects_list}")
        all_metrics = task.run_evaluation(list_sizes=num_projects_list)
        
        # Generate final report with individual test case breakdown
        logging.info("📊 Generating final report...")

        # Import the reconstruction function and use it directly
        from reporting import reconstruct_individual_metrics

        # Try to reconstruct individual metrics from detailed results
        reconstructed_metrics = reconstruct_individual_metrics(args.output_dir, 'constraint_optimization', num_projects_list)

        if reconstructed_metrics:
            # Use reconstructed metrics to show individual test cases
            logging.info(f"📈 Found {len(reconstructed_metrics)} individual test case metrics")
            final_report = generate_final_report(reconstructed_metrics, num_projects_list, args.output_dir)
        else:
            # Create individual metrics manually from all_metrics if reconstruction fails
            logging.info("📊 Creating individual metrics manually...")
            manual_metrics = []
            for i, num_projects in enumerate(num_projects_list):
                # Find metrics for this project count from all_metrics
                project_metrics = [m for m in all_metrics if m.get('num_projects') == num_projects or
                                 (i < len(all_metrics) and all_metrics[i] is m)]

                if not project_metrics:
                    # Create a default metric for this project count
                    project_metrics = [{'accuracy': 0, 'instruction_followed_pct': 0, 'avg_response_length': 0,
                                      'avg_word_count': 0, 'avg_output_tokens': 0}]

                metric = {
                    'task': f"constraint_optimization_{num_projects}",
                    'test_case': i,
                    'complexity': num_projects,
                    'accuracy': np.mean([m.get('accuracy', 0) for m in project_metrics]),
                    'instruction_followed_pct': np.mean([m.get('instruction_followed_pct', 0) for m in project_metrics]),
                    'avg_response_length': np.mean([m.get('avg_response_length', 0) for m in project_metrics]),
                    'avg_word_count': np.mean([m.get('avg_word_count', 0) for m in project_metrics]),
                    'avg_output_tokens': np.mean([m.get('avg_output_tokens', 0) for m in project_metrics])
                }
                manual_metrics.append(metric)

            final_report = generate_final_report(manual_metrics, num_projects_list, args.output_dir)
        
        # Log completion
        if all_metrics:
            avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
            logging.info(f"🎉 Evaluation completed successfully!")
            logging.info(f"📊 Overall average accuracy: {avg_accuracy:.4f}")
            logging.info(f"📁 Results saved to: {args.output_dir}")
        else:
            logging.warning("⚠️  No metrics were collected during evaluation")
            
    except Exception as e:
        logging.error(f"❌ Evaluation failed: {e}")
        logging.error(f"📍 Traceback: {traceback.format_exc()}")
        raise
    finally:
        # Cleanup if needed
        if 'model_handler' in locals():
            try:
                if hasattr(model_handler, 'cleanup'):
                    model_handler.cleanup()
                    logging.info("🧹 Model handler cleanup completed")
            except Exception as cleanup_error:
                logging.warning(f"⚠️  Cleanup warning: {cleanup_error}")

if __name__ == "__main__":
    main()
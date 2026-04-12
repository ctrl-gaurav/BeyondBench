"""
Enhanced Graph Coloring Problem Solving Task

This task evaluates LLM's ability to solve complex graph coloring problems through advanced constraint satisfaction reasoning.
Graph coloring involves assigning colors to vertices of a graph such that no two adjacent vertices have the same color.
The goal is to find a valid coloring using the minimum number of colors (chromatic number) or a specified number of colors.

ENHANCEMENTS:
- Complex graph types: planar, wheel, grid, petersen_variant, ramsey, dense_random (in addition to cycle, complete, tree, bipartite)
- Larger vertex counts: 8-28 vertices (increased from 4-6)
- Higher chromatic numbers: up to 6+ colors required (vs previous 2-3)
- Enhanced parsing: 20+ parsing patterns to handle diverse response formats
- Strategic prompting: Graph-type specific hints and verification checklists
- Robust validation: Accepts near-optimal solutions for complex graph types

Algorithm: Generate graphs with known chromatic numbers, ask for valid vertex colorings using k colors.
Reasoning: Requires advanced constraint satisfaction, graph structure analysis, conflict detection, and systematic optimization.

Example: For a wheel graph (8 vertices), minimum 4 colors needed:
Hub vertex connected to 7-cycle, requiring strategic color planning to avoid conflicts.

CLI USAGE:
python graph_coloring_task.py --model_id "Qwen/Qwen2.5-3B-Instruct" --datapoints 20 --folds 1 --graph_types "planar,wheel,grid,petersen_variant,ramsey,dense_random" --num_vertices 8,12,16,20,24,28 --temperature 0.1 --top_p 0.9 --max_tokens 8192 --seed 42 --tensor_parallel_size 1 --gpu_memory_utilization 0.96 --trust_remote_code False --store_details True
"""

# ============================================================================
# CONFIGURATION PARAMETERS (Customize as needed)
# ============================================================================

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TASKS = ["graph_coloring"]
# Engine selection - choose 'vllm' or 'transformers'
ENGINE = "vllm"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.96
TRUST_REMOTE_CODE = False

# Evaluation Configuration
DATAPOINTS = 20  # Number of samples per fold
FOLDS = 1  # Number of evaluation folds
GRAPH_TYPES = ["planar", "wheel", "grid", "petersen_variant", "ramsey", "dense_random", "cycle", "complete"]  # Types of graphs to generate
NUM_VERTICES = [8, 12, 16, 20, 24, 28]  # Number of vertices in graphs
STORE_DETAILS = True  # Save detailed results in JSON
SEED = 42  # Random seed for reproducibility

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


from ...core.base_task import BaseTask
from ...models.model_handler import ModelHandler
from ...utils.report_generator import generate_final_report
from ...utils.logging_utils import setup_logging

class GraphGenerator:
    """Generate different types of graphs with known properties"""
    
    @staticmethod
    def generate_complete_graph(n: int) -> Tuple[List[Tuple[int, int]], int]:
        """Generate complete graph K_n (chromatic number = n)"""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges, n
    
    @staticmethod
    def generate_cycle_graph(n: int) -> Tuple[List[Tuple[int, int]], int]:
        """Generate cycle graph C_n (chromatic number = 3 if n>2 and odd, 2 if n>2 and even)"""
        if n < 3:
            return [], 1
        
        edges = []
        for i in range(n):
            edges.append((i, (i + 1) % n))
        
        chromatic_number = 3 if n % 2 == 1 else 2
        return edges, chromatic_number
    
    @staticmethod
    def generate_tree_graph(n: int) -> Tuple[List[Tuple[int, int]], int]:
        """Generate a random tree (chromatic number = 2 if n > 1, 1 if n = 1)"""
        if n <= 1:
            return [], 1
        
        edges = []
        # Generate random tree by connecting each vertex to a previous one
        for i in range(1, n):
            parent = random.randint(0, i - 1)
            edges.append((parent, i))
        
        return edges, 2
    
    @staticmethod
    def generate_bipartite_graph(n: int) -> Tuple[List[Tuple[int, int]], int]:
        """Generate a random bipartite graph (chromatic number = 2 if has edges, 1 if no edges)"""
        if n < 2:
            return [], 1
        
        # Split vertices into two sets
        set1_size = n // 2
        set2_size = n - set1_size
        
        edges = []
        # Add some random edges between the two sets
        num_edges = random.randint(1, min(set1_size * set2_size, n))
        
        for _ in range(num_edges):
            v1 = random.randint(0, set1_size - 1)
            v2 = random.randint(set1_size, n - 1)
            if (v1, v2) not in edges and (v2, v1) not in edges:
                edges.append((v1, v2))
        
        chromatic_number = 2 if edges else 1
        return edges, chromatic_number
    
    @staticmethod
    def generate_planar_graph(n: int) -> Tuple[List[Tuple[int, int]], int]:
        """Generate a challenging planar graph with high chromatic number"""
        if n < 6:
            return GraphGenerator.generate_cycle_graph(n)
        
        edges = []
        # Create a core structure that requires 4 colors (like a K4 subdivision)
        if n >= 6:
            # Create two triangles connected by bridges
            # Triangle 1: 0-1-2-0
            edges.extend([(0, 1), (1, 2), (2, 0)])
            # Triangle 2: 3-4-5-3
            edges.extend([(3, 4), (4, 5), (5, 3)])
            
            # Connect the triangles to force 4-coloring
            if n >= 8:
                edges.extend([(0, 3), (1, 4), (2, 5)])
                
                # Add more vertices in a challenging pattern
                for i in range(6, min(n, 8)):
                    # Connect to create dependencies
                    edges.extend([(i, (i-6) % 6), (i, (i-5) % 6)])
                
                # For larger graphs, add structured complexity
                for i in range(8, n):
                    # Connect to multiple previous vertices to increase constraint complexity
                    targets = [i - 1, i - 2, (i * 3) % (i - 1), (i * 7) % (i - 2)]
                    for target in targets[:min(3, i)]:
                        if target >= 0 and target < i:
                            edges.append((target, i))
            else:
                # For n=6,7 connect triangles more simply
                edges.extend([(0, 3), (1, 4)])
                if n == 7:
                    edges.extend([(6, 0), (6, 3)])
        
        return edges, 4  # Most planar graphs can be 4-colored, but these require it
    
    @staticmethod
    def generate_wheel_graph(n: int) -> Tuple[List[Tuple[int, int]], int]:
        """Generate wheel graph W_n (hub + cycle, chromatic number = 3 or 4)"""
        if n < 4:
            return GraphGenerator.generate_complete_graph(n)
        
        edges = []
        # Create outer cycle (vertices 1 to n-1)
        for i in range(1, n-1):
            edges.append((i, i + 1))
        edges.append((n-1, 1))  # Close the cycle
        
        # Connect hub (vertex 0) to all vertices in cycle
        for i in range(1, n):
            edges.append((0, i))
        
        # Chromatic number is 3 if outer cycle is even, 4 if odd
        chromatic_number = 4 if (n - 1) % 2 == 1 else 3
        return edges, chromatic_number
    
    @staticmethod
    def generate_grid_graph(n: int) -> Tuple[List[Tuple[int, int]], int]:
        """Generate grid graph (chromatic number = 2 for most, but can be challenging to find)"""
        if n < 4:
            return GraphGenerator.generate_cycle_graph(n)
        
        # Create as close to square grid as possible
        rows = int(n**0.5)
        cols = (n + rows - 1) // rows
        
        edges = []
        vertex_map = {}
        vertex_count = 0
        
        # Create vertex mapping
        for r in range(rows):
            for c in range(cols):
                if vertex_count < n:
                    vertex_map[(r, c)] = vertex_count
                    vertex_count += 1
        
        # Add edges
        for r in range(rows):
            for c in range(cols):
                if (r, c) in vertex_map:
                    current = vertex_map[(r, c)]
                    # Right neighbor
                    if (r, c + 1) in vertex_map:
                        edges.append((current, vertex_map[(r, c + 1)]))
                    # Down neighbor
                    if (r + 1, c) in vertex_map:
                        edges.append((current, vertex_map[(r + 1, c)]))
        
        return edges, 2  # Grid graphs are 2-colorable but can be tricky to visualize
    
    @staticmethod
    def generate_petersen_variant(n: int) -> Tuple[List[Tuple[int, int]], int]:
        """Generate Petersen-like graph variant (chromatic number = 3)"""
        if n < 6:
            return GraphGenerator.generate_complete_graph(min(n, 3))
        
        edges = []
        
        # For Petersen-like structure, we need at least 10 vertices for the classic
        # For smaller n, create a challenging 3-chromatic structure
        if n < 10:
            # Create two triangles with connecting edges
            edges.extend([(0, 1), (1, 2), (2, 0)])  # Triangle 1
            if n >= 6:
                edges.extend([(3, 4), (4, 5), (5, 3)])  # Triangle 2
                edges.extend([(0, 3), (1, 4), (2, 5)])  # Connect them
            
            # Add remaining vertices with strategic connections
            for i in range(6, n):
                # Connect to force 3-coloring
                edges.extend([(i, i % 3), (i, (i + 1) % 3)])
        
        else:
            # Create classic Petersen-like structure
            # Outer pentagon: 0-1-2-3-4-0
            for i in range(5):
                edges.append((i, (i + 1) % 5))
            
            # Inner star: 5-6-7-8-9 with each connected to (i-5)*2 % 5
            for i in range(5, 10):
                outer_vertex = ((i - 5) * 2) % 5
                edges.append((outer_vertex, i))
            
            # Connect inner vertices in a pentagram
            for i in range(5, 10):
                edges.append((i, 5 + ((i - 5 + 2) % 5)))
            
            # Add remaining vertices with complex connections
            for i in range(10, n):
                # Connect to create dependencies forcing 3-coloring
                targets = [i % 10, (i * 3) % 10, (i * 7) % 10]
                for target in targets[:2]:
                    edges.append((target, i))
        
        return edges, 3
    
    @staticmethod
    def generate_ramsey_graph(n: int) -> Tuple[List[Tuple[int, int]], int]:
        """Generate Ramsey-type graph (designed to avoid large cliques/independent sets)"""
        if n < 6:
            return GraphGenerator.generate_cycle_graph(n)
        
        edges = []
        
        # Create a structure that avoids large cliques but requires many colors
        # Use a construction based on quadratic residues for pseudo-randomness
        for i in range(n):
            for j in range(i + 1, n):
                # Add edge based on a pattern that creates high chromatic number
                diff = j - i
                # Add edge if difference has certain properties
                if (diff * diff) % n in range(1, min(n//2 + 2, n)):
                    edges.append((i, j))
        
        # Ensure connectivity and estimate chromatic number
        # This construction typically gives chromatic number around sqrt(n)
        chromatic_number = max(3, min(int(n**0.6) + 1, n))
        
        return edges, chromatic_number
    
    @staticmethod
    def generate_dense_random_graph(n: int) -> Tuple[List[Tuple[int, int]], int]:
        """Generate dense random graph with high chromatic number"""
        if n < 4:
            return GraphGenerator.generate_complete_graph(n)

        edges = []
        edge_probability = 0.6  # High density to force many colors

        # Use random module for seed-dependent generation
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < edge_probability:
                    edges.append((i, j))

        # Estimate chromatic number based on clique number and density
        # For dense random graphs, chromatic number is typically high
        chromatic_number = max(3, min(int(n * edge_probability * 0.8) + 1, n))

        return edges, chromatic_number

class GraphColoringSolver:
    """Ground truth solver for graph coloring problems"""
    
    @staticmethod
    def is_valid_coloring(edges: List[Tuple[int, int]], coloring: Dict[int, int], n: int) -> Tuple[bool, str]:
        """Check if a coloring is valid"""
        # Check if all vertices are colored
        for v in range(n):
            if v not in coloring:
                return False, f"Vertex {v} is not colored"
        
        # Check if adjacent vertices have different colors
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False, f"Adjacent vertices {u} and {v} have the same color {coloring[u]}"
        
        return True, "Valid coloring"
    
    @staticmethod
    def solve_graph_coloring(edges: List[Tuple[int, int]], n: int, max_colors: int) -> Optional[Dict[int, int]]:
        """Find a valid coloring using at most max_colors colors using backtracking"""
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        coloring = {}
        
        def backtrack(vertex: int) -> bool:
            if vertex == n:
                return True  # All vertices colored
            
            # Try each color
            for color in range(max_colors):
                # Check if this color conflicts with neighbors
                valid = True
                for neighbor in adj[vertex]:
                    if neighbor in coloring and coloring[neighbor] == color:
                        valid = False
                        break
                
                if valid:
                    coloring[vertex] = color
                    if backtrack(vertex + 1):
                        return True
                    del coloring[vertex]
            
            return False
        
        if backtrack(0):
            return coloring
        return None

class GraphColoringTask(BaseTask):
    """Graph Coloring evaluation task with robust parsing"""
    
    def __init__(self, model_handler, output_dir, graph_types, num_vertices_list, num_folds, 
                 num_samples, store_details, temperature, top_p, max_tokens, seed=None):
        # Initialize base class with proper min/max values
        self._task_name = "graph_coloring"
        super().__init__(model_handler, output_dir, min(num_vertices_list), max(num_vertices_list), num_folds, num_samples, 
                        store_details, temperature, top_p, max_tokens, seed)
        self.graph_types = graph_types
        self.num_vertices_list = num_vertices_list
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Enhanced parsing patterns with confidence scoring
        self.setup_parsing_patterns()
    
    @property
    def task_name(self):
        """Return task name"""
        return self._task_name
    
    def generate_data(self, list_size=None, **kwargs):
        """Generate evaluation data
        
        Args:
            list_size: Number of vertices (4, 5, 6, etc.)
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        num_vertices = list_size if list_size is not None else random.choice(self.num_vertices_list)
        test_cases = []
        
        # Generate exactly num_samples total, distributed across graph types
        samples_per_type = max(1, self.num_samples // len(self.graph_types))
        remaining_samples = self.num_samples - (samples_per_type * len(self.graph_types))
        
        for i, graph_type in enumerate(self.graph_types):
            # Add extra sample to first types if needed to reach exact num_samples
            samples_for_this_type = samples_per_type + (1 if i < remaining_samples else 0)
            
            for _ in range(samples_for_this_type):
                problem = self.generate_problem(graph_type, num_vertices)
                if problem['reference_solution'] is not None:
                    problem['answer'] = problem['reference_solution']  # For BaseTask compatibility
                    test_cases.append(problem)
        
        # Ensure we have exactly num_samples by padding if necessary
        while len(test_cases) < self.num_samples:
            graph_type = random.choice(self.graph_types)
            problem = self.generate_problem(graph_type, num_vertices)
            if problem['reference_solution'] is not None:
                problem['answer'] = problem['reference_solution']
                test_cases.append(problem)
        
        return test_cases[:self.num_samples]  # Ensure exact count
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response with robust handling"""
        result = self.parse_response(response, data_point)
        
        # Always include parsed result even if invalid, like N-Queens implementation
        parsed_coloring = result.get('parsed_coloring', {})
        is_correct = result.get('is_correct', False)
        parsing_confidence = result.get('parsing_confidence', 0.0)
        
        # More lenient instruction following - consider successful if we parsed something
        instruction_followed = 1 if parsed_coloring or parsing_confidence > 0.3 else 0
        
        return {
            'accuracy': 1 if is_correct else 0,
            'instruction_followed': instruction_followed,
            'predicted_answer': parsed_coloring,  # Always include what was parsed
            'ground_truth': data_point.get('reference_solution', {}),
            'parsing_info': {
                'method': result.get('parsing_method', 'unknown'),
                'confidence': parsing_confidence,
                'attempts': result.get('parsing_attempts', []),
                'raw_response_preview': response[:200] + '...' if len(response) > 200 else response
            },
            'validation_info': result.get('validation_info', {})
        }
    
    def setup_parsing_patterns(self):
        """Setup regex patterns for parsing different coloring formats"""
        self.coloring_patterns = [
            # Pattern 1: Standard dictionary format {0: "Red", 1: "Blue"}
            (re.compile(r'\{\s*([^}]+)\s*\}', re.IGNORECASE), "dict_format", 0.95),
            
            # Pattern 2: Vertex-color pairs format "vertex 0 color Red, vertex 1 color Blue"
            (re.compile(r'vertex\s+(\d+)\s+(?:is\s+)?color\s+([\w"\']+)', re.IGNORECASE), "vertex_color_pairs", 0.85),
            
            # Pattern 3: Assignment format "0 -> Red, 1 -> Blue" or "0: Red, 1: Blue"
            (re.compile(r'(\d+)\s*(?:->|:|=)\s*([\w"\']+)', re.IGNORECASE), "assignment_format", 0.88),
            
            # Pattern 4: List format with labels "vertex: [0, 1, 2], color: [Red, Blue, Green]"
            (re.compile(r'vertex(?:es)?\s*:\s*\[([^\]]+)\]\s*,?\s*colors?\s*:\s*\[([^\]]+)\]', re.IGNORECASE), "parallel_lists", 0.80),
            
            # Pattern 5: Numbered color format {0: 0, 1: 1, 2: 2}
            (re.compile(r'\{\s*([0-9,:;\s]+)\s*\}', re.IGNORECASE), "numeric_dict", 0.82),
            
            # Pattern 6: Simple sequence "Red Blue Green" or "0 1 2" (only if exactly n_vertices words)
            (re.compile(r'([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)', re.IGNORECASE), "sequence_format", 0.50),
            
            # Pattern 7: Color-vertex mapping "Red: 0, Blue: 1, Green: 2"
            (re.compile(r'([\w"\']+)\s*:\s*(\d+)', re.IGNORECASE), "color_vertex_mapping", 0.65),
            
            # Pattern 8: Brackets with assignments [0=Red, 1=Blue, 2=Green]
            (re.compile(r'\[([^\]]+)\]', re.IGNORECASE), "bracket_assignments", 0.75),
            
            # Pattern 9: Tabular format "0 | Red\n1 | Blue"
            (re.compile(r'(\d+)\s*[\|\-]\s*([\w"\']+)', re.IGNORECASE), "table_format", 0.70),
            
            # Pattern 10: Solution keyword "Solution: {0: Red, 1: Blue}"
            (re.compile(r'(?:solution|answer|result|coloring|final)\s*:?\s*(.+?)(?:\n|$)', re.IGNORECASE), "solution_keyword", 0.85),
            
            # NEW: Pattern 17: Final solution format
            (re.compile(r'final\s+solution\s*:?\s*(.+?)(?:\n|$)', re.IGNORECASE), "final_solution", 0.90),
            
            # NEW: Pattern 11: Node format "node 0: red, node 1: blue"
            (re.compile(r'node\s+(\d+)\s*:\s*([\w"\']+)', re.IGNORECASE), "node_format", 0.83),
            
            # NEW: Pattern 12: V format "v0=red, v1=blue"
            (re.compile(r'v(\d+)\s*=\s*([\w"\']+)', re.IGNORECASE), "v_format", 0.78),
            
            # NEW: Pattern 13: Parentheses format (0, "red"), (1, "blue")
            (re.compile(r'\(\s*(\d+)\s*,\s*["\']?([\w"\']+)["\']?\s*\)', re.IGNORECASE), "parentheses_pairs", 0.77),
            
            # NEW: Pattern 14: Mapping format "0 maps to red, 1 maps to blue"
            (re.compile(r'(\d+)\s+maps?\s+to\s+([\w"\']+)', re.IGNORECASE), "mapping_format", 0.72),
            
            # NEW: Pattern 15: Gets format "vertex 0 gets red, vertex 1 gets blue"
            (re.compile(r'vertex\s+(\d+)\s+gets\s+([\w"\']+)', re.IGNORECASE), "gets_format", 0.74),
            
            # NEW: Pattern 16: Has format "vertex 0 has red, vertex 1 has blue"
            (re.compile(r'vertex\s+(\d+)\s+has\s+([\w"\']+)', re.IGNORECASE), "has_format", 0.73),
            
            # NEW: Pattern 17: Color assignment format "assign red to 0, assign blue to 1"
            (re.compile(r'assign\s+([\w"\']+)\s+to\s+(\d+)', re.IGNORECASE), "assign_format", 0.76),
            
            # NEW: Pattern 18: Arrow format with colors "0 => red, 1 => blue"
            (re.compile(r'(\d+)\s*=>\s*([\w"\']+)', re.IGNORECASE), "arrow_format", 0.79),
            
            # NEW: Pattern 19: Colon separated list "0:red, 1:blue, 2:green"
            (re.compile(r'(\d+):\s*([\w"\']+)(?:\s*,\s*|\s*$)', re.IGNORECASE), "colon_list_format", 0.81),
            
            # NEW: Pattern 20: JSON-like but without braces "0: 'red', 1: 'blue'"
            (re.compile(r'(\d+):\s*[\'\"]([\w]+)[\'\"]\s*(?:,|$)', re.IGNORECASE), "json_like_quotes", 0.84)
        ]
    
    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for prompt"""
        return len(prompt) // 4
    
    def generate_problem(self, graph_type: str, n: int) -> Dict[str, Any]:
        """Generate a graph coloring problem"""
        # Generate graph based on type
        if graph_type == "complete":
            edges, chromatic_number = GraphGenerator.generate_complete_graph(n)
        elif graph_type == "cycle":
            edges, chromatic_number = GraphGenerator.generate_cycle_graph(n)
        elif graph_type == "tree":
            edges, chromatic_number = GraphGenerator.generate_tree_graph(n)
        elif graph_type == "bipartite":
            edges, chromatic_number = GraphGenerator.generate_bipartite_graph(n)
        elif graph_type == "planar":
            edges, chromatic_number = GraphGenerator.generate_planar_graph(n)
        elif graph_type == "wheel":
            edges, chromatic_number = GraphGenerator.generate_wheel_graph(n)
        elif graph_type == "grid":
            edges, chromatic_number = GraphGenerator.generate_grid_graph(n)
        elif graph_type == "petersen_variant":
            edges, chromatic_number = GraphGenerator.generate_petersen_variant(n)
        elif graph_type == "ramsey":
            edges, chromatic_number = GraphGenerator.generate_ramsey_graph(n)
        elif graph_type == "dense_random":
            edges, chromatic_number = GraphGenerator.generate_dense_random_graph(n)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        # Find a reference solution
        reference_solution = GraphColoringSolver.solve_graph_coloring(edges, n, chromatic_number)
        
        return {
            'n': n,
            'graph_type': graph_type,
            'edges': edges,
            'chromatic_number': chromatic_number,
            'reference_solution': reference_solution
        }
    
    def create_prompt(self, problem: Dict[str, Any]) -> str:
        """Create enhanced prompt for complex graph coloring problems"""
        n = problem['n']
        edges = problem['edges']
        chromatic_number = problem['chromatic_number']
        graph_type = problem.get('graph_type', 'unknown')
        
        edges_str = ', '.join([f"({u},{v})" for u, v in edges])
        
        # Add strategic hints based on graph type and complexity
        strategy_hint = ""
        if graph_type == "wheel":
            strategy_hint = "\n\nSTRATEGY HINT: This is a wheel graph. Consider the hub vertex and the cycle pattern carefully."
        elif graph_type == "planar":
            strategy_hint = "\n\nSTRATEGY HINT: This is a planar graph. Look for triangular substructures that may constrain coloring choices."
        elif graph_type == "grid":
            strategy_hint = "\n\nSTRATEGY HINT: This graph has a grid-like structure. Consider the regularity of connections."
        elif graph_type in ["ramsey", "dense_random"]:
            strategy_hint = "\n\nSTRATEGY HINT: This graph has complex connectivity. Carefully analyze which vertices are adjacent before assigning colors."
        elif chromatic_number >= 4:
            strategy_hint = "\n\nSTRATEGY HINT: This graph requires many colors. Look for highly connected subgraphs (cliques) that force different colors."
        
        # Enhanced constraint explanation for higher chromatic numbers
        constraint_explanation = ""
        if chromatic_number >= 4:
            constraint_explanation = f"""\n\nCONSTRAINT ANALYSIS REQUIRED:
- This problem requires {chromatic_number} different colors - more than typical simple graphs
- Look for vertices that are connected to many other vertices
- Identify groups of vertices that are all connected to each other (cliques)
- Each clique of size k requires k different colors
- Plan your color assignments carefully to avoid conflicts"""
        
        # Add verification checklist for complex problems
        verification_checklist = f"""\n\nVERIFICATION CHECKLIST:
1. Count your colors: Did you use exactly {chromatic_number} different colors?
2. Check ALL edges: For each edge ({', '.join([f'{u}-{v}' for u, v in edges[:5]])}{'...' if len(edges) > 5 else ''}), verify adjacent vertices have different colors
3. Complete assignment: Did you assign a color to all {n} vertices (0, 1, 2, ..., {n-1})?
4. No conflicts: Double-check that no two adjacent vertices share the same color"""
        
        prompt = f"""CHALLENGING GRAPH COLORING PROBLEM:

Your task: Color the vertices of a complex graph using exactly {chromatic_number} colors such that no two adjacent (connected) vertices have the same color.

GRAPH SPECIFICATION:
- Graph type: {graph_type}
- {n} vertices numbered from 0 to {n-1}
- {len(edges)} edges: {edges_str}
- Minimum colors needed: {chromatic_number}{strategy_hint}{constraint_explanation}

COLORING RULES:
- Use exactly {chromatic_number} different colors (you can use color names like Red, Blue, Green, Yellow, Orange, Purple, etc., or numbers like 0, 1, 2, 3, ...)
- NO two vertices connected by an edge can have the same color
- EVERY vertex must be assigned exactly one color
- Each vertex (0 through {n-1}) needs exactly one color assignment

CRITICAL REQUIREMENTS:
- Your solution MUST assign a color to ALL {n} vertices (0, 1, 2, ..., {n-1})
- Use ALL {chromatic_number} colors (no fewer, no more)
- Verify no adjacent vertices have the same color

SOLUTION FORMAT REQUIREMENTS:
Provide your solution as a complete dictionary mapping each vertex number to its color.

ACCEPTED FORMATS:
- {{0: "Red", 1: "Blue", 2: "Green", 3: "Yellow", ...}} ← STRONGLY PREFERRED
- {{0: 0, 1: 1, 2: 2, 3: 3, ...}}
- 0: Red, 1: Blue, 2: Green, 3: Yellow, ...

EXAMPLE for 4 vertices with 3 colors:
If vertex 0→Red, vertex 1→Blue, vertex 2→Green, vertex 3→Blue:
Answer: {{0: "Red", 1: "Blue", 2: "Green", 3: "Blue"}}{verification_checklist}

FINAL ANSWER FORMAT:
End your response with the complete solution dictionary in this format:

<answer>
{{0: "ColorA", 1: "ColorB", 2: "ColorC", ..., {n-1}: "ColorX"}}
</answer>

ALTERNATIVELY, clearly mark your final dictionary as "Final solution:" or "Answer:".

SOLVING APPROACH:
1. Analyze the graph structure and identify highly connected vertices
2. Start coloring vertices with the most constraints (highest degree)
3. Work systematically through all vertices
4. Verify your solution against all edges
5. Double-check you used exactly {chromatic_number} colors

Provide the complete dictionary with all {n} vertex assignments."""
        
        return prompt
    
    def parse_response(self, response: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response with enhanced robustness and detailed diagnostics"""
        n_vertices = problem['n']
        parsing_attempts = []
        best_coloring = {}
        best_confidence = 0.0
        best_method = "none"
        
        # Extract answer with priority extraction like N-Queens
        answer_text = response.strip()
        
        # Priority extraction order
        if '<answer>' in response and '</answer>' in response:
            answer_text = response.split('<answer>')[1].split('</answer>')[0].strip()
            parsing_attempts.append(("answer_tags", "Extracted from <answer> tags", 0.1))
        elif '```' in response:
            # Extract from code blocks - try all blocks, prefer ones with dictionary-like content
            code_blocks = re.findall(r'```(?:[a-z]*\n)?(.*?)```', response, re.DOTALL)
            if code_blocks:
                # Look for the best code block (one that contains dictionary-like structure)
                best_block = None
                best_score = -1
                
                for block in code_blocks:
                    block = block.strip()
                    # Score based on how likely this contains the final answer
                    score = 0
                    if '{' in block and '}' in block:
                        score += 2
                    if 'Color' in block or '"' in block:
                        score += 2
                    if block.count(':') > 5:  # Likely a dictionary with many entries
                        score += 3
                    if 'print' not in block and 'for' not in block:  # Avoid intermediate code
                        score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_block = block
                
                answer_text = best_block if best_block else code_blocks[-1].strip()  # Use last if no clear winner
                parsing_attempts.append(("code_blocks", f"Extracted from code blocks (score: {best_score})", 0.05))
        elif 'ANSWER:' in response.upper():
            # Extract after "ANSWER:" (case-insensitive)
            parts = re.split(r'answer\s*:', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer_text = parts[1].strip()
                parsing_attempts.append(("answer_keyword", "Extracted after ANSWER:", 0.05))
        elif 'SOLUTION:' in response.upper():
            # Extract after "SOLUTION:" (case-insensitive)
            parts = re.split(r'solution\s*:', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer_text = parts[1].strip()
                parsing_attempts.append(("solution_keyword", "Extracted after SOLUTION:", 0.05))
        
        # Try each parsing pattern
        for pattern, method_name, base_confidence in self.coloring_patterns:
            try:
                coloring = self.parse_with_pattern(answer_text, pattern, method_name, n_vertices)
                if coloring and len(coloring) == n_vertices:
                    confidence = base_confidence * (len(coloring) / n_vertices)
                    parsing_attempts.append((method_name, f"Parsed {len(coloring)} vertices", confidence))
                    
                    if confidence > best_confidence:
                        best_coloring = coloring
                        best_confidence = confidence
                        best_method = method_name
                elif coloring:
                    parsing_attempts.append((method_name, f"Partial parse: {len(coloring)}/{n_vertices} vertices", base_confidence * 0.3))
                else:
                    parsing_attempts.append((method_name, "No match", 0.0))
                    
            except Exception as e:
                parsing_attempts.append((method_name, f"Error: {str(e)}", 0.0))
        
        # Fallback: try to find any dictionary-like structure
        if not best_coloring:
            # Last resort: find any curly braces and try to parse
            brace_matches = re.findall(r'\{[^}]*\}', answer_text)
            for match in brace_matches:
                try:
                    fallback_coloring = self.parse_dict_content(match)
                    if len(fallback_coloring) == n_vertices:
                        parsing_attempts.append(("fallback_braces", f"Found dict: {match[:50]}...", 0.3))
                        if 0.3 > best_confidence:
                            best_coloring = fallback_coloring
                            best_confidence = 0.3
                            best_method = "fallback_braces"
                        break
                except:
                    continue
            
            # Even more desperate fallback: try to find vertex:color patterns anywhere
            if not best_coloring:
                fallback_patterns = [
                    (r'(\d+)\s*[=:]\s*([\w"\']+)', "fallback_assignment", 0.2),
                    (r'vertex\s+(\d+)[^\d]*?([\w"\']+)', "fallback_vertex", 0.15),
                ]
                
                # Super fallback: look for any large dictionary in the entire response
                if not best_coloring:
                    # Find all dictionaries in the response, prefer the largest one
                    all_dicts = re.findall(r'\{[^}]{50,}\}', response)  # Dictionaries with 50+ chars
                    for dict_match in all_dicts:
                        try:
                            fallback_coloring = self.parse_dict_content(dict_match)
                            if len(fallback_coloring) == n_vertices:
                                parsing_attempts.append(("super_fallback_dict", f"Found large dict: {len(fallback_coloring)} vertices", 0.4))
                                if 0.4 > best_confidence:
                                    best_coloring = fallback_coloring
                                    best_confidence = 0.4
                                    best_method = "super_fallback_dict"
                                break
                        except:
                            continue
                
                for pattern_str, name, conf in fallback_patterns:
                    try:
                        pattern = re.compile(pattern_str, re.IGNORECASE)
                        matches = pattern.findall(answer_text)
                        if len(matches) == n_vertices:
                            fallback_coloring = {}
                            color_map = {}
                            next_color_id = 0
                            
                            for vertex_str, color in matches:
                                vertex = int(vertex_str)
                                if 0 <= vertex < n_vertices:
                                    clean_color = color.strip('"\' \\t\\n')
                                    if clean_color not in color_map:
                                        color_map[clean_color] = next_color_id
                                        next_color_id += 1
                                    fallback_coloring[vertex] = color_map[clean_color]
                            
                            if len(fallback_coloring) == n_vertices:
                                parsing_attempts.append((name, f"Found {len(matches)} matches", conf))
                                if conf > best_confidence:
                                    best_coloring = fallback_coloring
                                    best_confidence = conf
                                    best_method = name
                                break
                    except:
                        continue
        
        # Validate the best solution
        validation_info = {}
        if best_coloring:
            is_valid, validation_msg = GraphColoringSolver.is_valid_coloring(
                problem['edges'], best_coloring, n_vertices
            )
            
            colors_used = len(set(best_coloring.values()))
            expected_colors = problem['chromatic_number']
            
            # Check completeness
            missing_vertices = [v for v in range(n_vertices) if v not in best_coloring]
            if missing_vertices:
                is_valid = False
                validation_msg += f" Missing vertices: {missing_vertices}"
            
            # Check color efficiency - be more lenient for complex graphs
            # Allow using more colors if the solution is valid, especially for complex graph types
            graph_type = problem.get('graph_type', 'unknown')
            is_correct = is_valid
            
            # For complex graph types, accept solutions that use up to 1 extra color
            if graph_type in ["ramsey", "dense_random", "planar"] and colors_used <= expected_colors + 1:
                is_correct = is_valid
            elif colors_used <= expected_colors:
                is_correct = is_valid
            else:
                is_correct = False
                
            if is_valid and colors_used > expected_colors:
                excess = colors_used - expected_colors
                if excess == 1 and graph_type in ["ramsey", "dense_random", "planar"]:
                    validation_msg += f" Uses {colors_used} colors (optimal: {expected_colors}, +1 acceptable for {graph_type})"
                else:
                    validation_msg += f" Uses {colors_used} colors (optimal: {expected_colors})"
            
            validation_info = {
                'is_valid_coloring': is_valid,
                'colors_used': colors_used,
                'expected_colors': expected_colors,
                'is_optimal': colors_used <= expected_colors,
                'missing_vertices': missing_vertices,
                'validation_message': validation_msg
            }
        else:
            is_correct = False
            validation_info = {
                'is_valid_coloring': False,
                'colors_used': 0,
                'expected_colors': problem['chromatic_number'],
                'is_optimal': False,
                'missing_vertices': list(range(n_vertices)),
                'validation_message': "Could not parse any coloring from response"
            }
        
        return {
            'parsed_coloring': best_coloring,
            'parsing_method': best_method,
            'parsing_confidence': best_confidence,
            'parsing_attempts': parsing_attempts,
            'validation_info': validation_info,
            'is_correct': is_correct,
            'raw_response': response
        }
    
    def parse_with_pattern(self, text: str, pattern: re.Pattern, method_name: str, n_vertices: int) -> Dict[int, int]:
        """Parse coloring using a specific pattern"""
        coloring = {}
        
        if method_name == "dict_format":
            # Standard dictionary format {0: "Red", 1: "Blue"}
            matches = pattern.findall(text)
            if matches:
                dict_content = matches[0]
                return self.parse_dict_content(dict_content)
        
        elif method_name == "vertex_color_pairs":
            # Vertex-color pairs format
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    if color not in color_map:
                        color_map[color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[color]
        
        elif method_name == "assignment_format":
            # Assignment format "0 -> Red, 1 -> Blue"
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    if color not in color_map:
                        color_map[color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[color]
        
        elif method_name == "parallel_lists":
            # List format "vertex: [0, 1, 2], color: [Red, Blue, Green]"
            matches = pattern.findall(text)
            if matches:
                vertices_str, colors_str = matches[0]
                vertices = [int(v.strip()) for v in vertices_str.split(',')]
                colors = [c.strip().strip('"\'') for c in colors_str.split(',')]
                
                if len(vertices) == len(colors):
                    color_map = {}
                    next_color_id = 0
                    
                    for vertex, color in zip(vertices, colors):
                        if 0 <= vertex < n_vertices:
                            if color not in color_map:
                                color_map[color] = next_color_id
                                next_color_id += 1
                            coloring[vertex] = color_map[color]
        
        elif method_name == "numeric_dict":
            # Numeric dictionary {0: 0, 1: 1}
            content = pattern.findall(text)
            if content:
                pairs = re.findall(r'(\d+)\s*[,:;]\s*(\d+)', content[0])
                for vertex_str, color_str in pairs:
                    vertex = int(vertex_str)
                    color = int(color_str)
                    if 0 <= vertex < n_vertices:
                        coloring[vertex] = color
        
        elif method_name == "sequence_format":
            # Simple sequence "Red Blue Green"
            matches = pattern.findall(text)
            if matches:
                colors = matches[0].split()
                if len(colors) == n_vertices:
                    color_map = {}
                    next_color_id = 0
                    
                    for vertex, color in enumerate(colors):
                        if color not in color_map:
                            color_map[color] = next_color_id
                            next_color_id += 1
                        coloring[vertex] = color_map[color]
        
        elif method_name == "color_vertex_mapping":
            # Color-vertex mapping "Red: 0, Blue: 1"
            matches = pattern.findall(text)
            color_to_vertices = {}
            
            for color, vertex_str in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    if color not in color_to_vertices:
                        color_to_vertices[color] = []
                    color_to_vertices[color].append(vertex)
            
            # Convert to vertex -> color_id mapping
            for color_id, (color, vertices) in enumerate(color_to_vertices.items()):
                for vertex in vertices:
                    coloring[vertex] = color_id
        
        elif method_name == "bracket_assignments":
            # Brackets with assignments [0=Red, 1=Blue]
            matches = pattern.findall(text)
            if matches:
                content = matches[0]
                assignments = re.findall(r'(\d+)\s*=\s*(\w+)', content)
                color_map = {}
                next_color_id = 0
                
                for vertex_str, color in assignments:
                    vertex = int(vertex_str)
                    if 0 <= vertex < n_vertices:
                        if color not in color_map:
                            color_map[color] = next_color_id
                            next_color_id += 1
                        coloring[vertex] = color_map[color]
        
        elif method_name == "table_format":
            # Tabular format "0 | Red"
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    if color not in color_map:
                        color_map[color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[color]
        
        elif method_name == "solution_keyword":
            # Solution keyword
            matches = pattern.findall(text)
            if matches:
                solution_text = matches[0].strip()
                # Recursively parse the solution text
                return self.parse_dict_content(solution_text)
        
        elif method_name == "node_format":
            # Node format "node 0: red, node 1: blue"
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    clean_color = color.strip('"\' \\t\\n')
                    if clean_color not in color_map:
                        color_map[clean_color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[clean_color]
        
        elif method_name == "v_format":
            # V format "v0=red, v1=blue"
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    clean_color = color.strip('"\' \\t\\n')
                    if clean_color not in color_map:
                        color_map[clean_color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[clean_color]
        
        elif method_name == "parentheses_pairs":
            # Parentheses format (0, "red"), (1, "blue")
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    clean_color = color.strip('"\' \\t\\n')
                    if clean_color not in color_map:
                        color_map[clean_color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[clean_color]
        
        elif method_name == "mapping_format":
            # Mapping format "0 maps to red, 1 maps to blue"
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    clean_color = color.strip('"\' \\t\\n')
                    if clean_color not in color_map:
                        color_map[clean_color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[clean_color]
        
        elif method_name in ["gets_format", "has_format"]:
            # Gets/Has format "vertex 0 gets red, vertex 1 has blue"
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    clean_color = color.strip('"\' \\t\\n')
                    if clean_color not in color_map:
                        color_map[clean_color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[clean_color]
        
        elif method_name == "assign_format":
            # Assign format "assign red to 0, assign blue to 1"
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for color, vertex_str in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    clean_color = color.strip('"\' \\t\\n')
                    if clean_color not in color_map:
                        color_map[clean_color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[clean_color]
        
        elif method_name in ["arrow_format", "colon_list_format"]:
            # Arrow or colon format "0 => red, 1 => blue" or "0:red, 1:blue"
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    clean_color = color.strip('"\' \\t\\n')
                    if clean_color not in color_map:
                        color_map[clean_color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[clean_color]
        
        elif method_name == "json_like_quotes":
            # JSON-like with quotes format "0: 'red', 1: 'blue'"
            matches = pattern.findall(text)
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color in matches:
                vertex = int(vertex_str)
                if 0 <= vertex < n_vertices:
                    clean_color = color.strip('"\' \\t\\n')
                    if clean_color not in color_map:
                        color_map[clean_color] = next_color_id
                        next_color_id += 1
                    coloring[vertex] = color_map[clean_color]
        
        return coloring
    
    def parse_dict_content(self, content: str) -> Dict[int, int]:
        """Parse dictionary-like content"""
        coloring = {}
        
        # Try ast.literal_eval first
        try:
            if content.startswith('{') and content.endswith('}'):
                raw_dict = ast.literal_eval(content)
            else:
                raw_dict = ast.literal_eval('{' + content + '}')
                
            if isinstance(raw_dict, dict):
                color_map = {}
                next_color_id = 0
                
                for vertex, color in raw_dict.items():
                    if isinstance(vertex, str):
                        vertex = int(vertex)
                    
                    if color not in color_map:
                        color_map[color] = next_color_id
                        next_color_id += 1
                    
                    coloring[vertex] = color_map[color]
        except:
            # Manual parsing as fallback - enhanced regex for better matching
            pairs = re.findall(r'(\d+)\s*[:=]\s*["\']?([^,}\]\)]+?)["\']?(?:\s*[,}]|$)', content)
            if not pairs:
                # Try alternative pattern for cases like: 0: "Color1", 1: "Color2"
                pairs = re.findall(r'(\d+)\s*:\s*["\']?([^,}\]]+?)["\']?(?=\s*,|\s*}|$)', content)
            
            color_map = {}
            next_color_id = 0
            
            for vertex_str, color_str in pairs:
                vertex = int(vertex_str)
                color = color_str.strip('"\' \t\n')
                
                if color not in color_map:
                    color_map[color] = next_color_id
                    next_color_id += 1
                
                coloring[vertex] = color_map[color]
        
        return coloring
    
    

def main():
    parser = argparse.ArgumentParser(description="Graph Coloring Problem Solving Evaluation")
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Model identifier')
    parser.add_argument('--engine', type=str, default=ENGINE, choices=['vllm', 'transformers'], help='Inference engine')
    parser.add_argument('--tensor_parallel_size', type=int, default=TENSOR_PARALLEL_SIZE, help='Tensor parallel size for VLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=GPU_MEMORY_UTILIZATION, help='GPU memory utilization for VLLM')
    parser.add_argument('--trust_remote_code', type=bool, default=TRUST_REMOTE_CODE, help='Trust remote code')
    
    # Task configuration
    parser.add_argument('--datapoints', type=int, default=DATAPOINTS, help='Number of datapoints per fold')
    parser.add_argument('--folds', type=int, default=FOLDS, help='Number of evaluation folds')
    parser.add_argument('--graph_types', type=str, default=','.join(GRAPH_TYPES), help='Graph types (comma-separated)')
    parser.add_argument('--num_vertices', type=str, default=','.join(map(str, NUM_VERTICES)), help='Number of vertices (comma-separated)')
    parser.add_argument('--store_details', type=bool, default=STORE_DETAILS, help='Store detailed results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=TOP_P, help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS, help='Maximum tokens to generate')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='graph_coloring_results', help='Output directory')

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
    graph_types = [x.strip() for x in args.graph_types.split(',')]
    num_vertices_list = [int(x.strip()) for x in args.num_vertices.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("🎨 Starting Graph Coloring Problem Solving Evaluation")
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
        logging.info(f"🎯 Initializing Graph Coloring task")
        task = GraphColoringTask(
            model_handler=model_handler,
            output_dir=args.output_dir,
            graph_types=graph_types,
            num_vertices_list=num_vertices_list,
            num_folds=args.folds,
            num_samples=args.datapoints,
            store_details=args.store_details,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed
        )
        
        # Now setup logging and reports in the task-specific directory
        task_log_file = setup_logging(task.task_dir)
        logging.info(f"📁 Task-specific directory: {task.task_dir}")
        logging.info(f"📝 Task log file: {task_log_file}")
    
        # Set random seeds
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            logging.info(f"🎲 Set random seed to {args.seed}")
    
        # Run evaluation
        logging.info(f"🚀 Starting evaluation with vertex counts: {num_vertices_list}")
        all_metrics = task.run_evaluation(list_sizes=num_vertices_list)
        
        # Generate final report with individual test case breakdown
        logging.info("📊 Generating final report...")

        # Import the reconstruction function and use it directly
        from reporting import reconstruct_individual_metrics

        # Try to reconstruct individual metrics from detailed results
        reconstructed_metrics = reconstruct_individual_metrics(task.task_dir, 'graph_coloring', num_vertices_list)

        if reconstructed_metrics:
            # Use reconstructed metrics to show individual test cases
            logging.info(f"📈 Found {len(reconstructed_metrics)} individual test case metrics")
            final_report = generate_final_report(reconstructed_metrics, num_vertices_list, task.task_dir)
        else:
            # Create individual metrics manually from all_metrics if reconstruction fails
            logging.info("📊 Creating individual metrics manually...")
            manual_metrics = []
            for i, num_vertices in enumerate(num_vertices_list):
                # Find metrics for this vertex count from all_metrics
                vertex_metrics = [m for m in all_metrics if m.get('num_vertices') == num_vertices or
                                (i < len(all_metrics) and all_metrics[i] is m)]

                if not vertex_metrics:
                    # Create a default metric for this vertex count
                    vertex_metrics = [{'accuracy': 0, 'instruction_followed_pct': 0, 'avg_response_length': 0,
                                     'avg_word_count': 0, 'avg_output_tokens': 0}]

                metric = {
                    'task': f"graph_coloring_{num_vertices}",
                    'test_case': i,
                    'complexity': num_vertices,
                    'accuracy': np.mean([m.get('accuracy', 0) for m in vertex_metrics]),
                    'instruction_followed_pct': np.mean([m.get('instruction_followed_pct', 0) for m in vertex_metrics]),
                    'avg_response_length': np.mean([m.get('avg_response_length', 0) for m in vertex_metrics]),
                    'avg_word_count': np.mean([m.get('avg_word_count', 0) for m in vertex_metrics]),
                    'avg_output_tokens': np.mean([m.get('avg_output_tokens', 0) for m in vertex_metrics])
                }
                manual_metrics.append(metric)

            final_report = generate_final_report(manual_metrics, num_vertices_list, task.task_dir)
        
        # Log completion
        if all_metrics:
            avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
            logging.info(f"🎉 Evaluation completed successfully!")
            logging.info(f"📊 Overall average accuracy: {avg_accuracy:.4f}")
            logging.info(f"📁 Results saved to: {task.task_dir}")
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
                # Model handler cleanup - if method exists
                if hasattr(model_handler, 'cleanup'):
                    model_handler.cleanup()
                    logging.info("🧹 Model handler cleanup completed")
            except Exception as cleanup_error:
                logging.warning(f"⚠️  Cleanup warning: {cleanup_error}")

if __name__ == "__main__":
    main()
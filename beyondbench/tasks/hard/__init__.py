"""Hard suite task implementations."""

# Import all task classes from the hard suite
try:
    from .boolean_sat_task import BooleanSATTask
    from .constraint_optimization_task import ConstraintOptimizationTask
    from .cryptarithmetic_task import CryptarithmeticTask
    from .graph_coloring_task import GraphColoringTask
    from .logic_grid_puzzles_task_enhanced import LogicGridPuzzlesTask
    from .matrix_chain_multiplication_task import MatrixChainTask
    from .modular_systems_solver_task import ModularSystemsTask
    from .n_queens_task import NQueensTask
    from .sudoku_task import SudokuTask
    from .tower_hanoi_task import RobustTowerHanoiTask

except ImportError as e:
    import logging
    logging.debug(f"Failed to import some hard task classes: {e}")

__all__ = [
    "BooleanSATTask",
    "ConstraintOptimizationTask",
    "CryptarithmeticTask",
    "GraphColoringTask",
    "LogicGridPuzzlesTask",
    "MatrixChainTask",
    "ModularSystemsTask",
    "NQueensTask",
    "SudokuTask",
    "RobustTowerHanoiTask"
]
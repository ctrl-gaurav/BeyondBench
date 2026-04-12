"""Centralized seed management for reproducible evaluations."""

import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_all_seeds(seed: Optional[int]) -> None:
    """Propagate seed to all random number generators.

    Sets seeds for:
    - random (Python stdlib)
    - numpy.random
    - torch (if available)
    - torch.cuda (if available)

    Also sets torch deterministic flags for reproducibility.

    Args:
        seed: Random seed. If None, no seeding is performed.
    """
    if seed is None:
        return

    # Python stdlib
    random.seed(seed)
    logger.info(f"Set random.seed({seed})")

    # numpy
    try:
        import numpy as np
        np.random.seed(seed)
        logger.info(f"Set numpy.random.seed({seed})")
    except ImportError:
        pass

    # torch
    try:
        import torch
        torch.manual_seed(seed)
        logger.info(f"Set torch.manual_seed({seed})")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            logger.info(f"Set torch.cuda.manual_seed_all({seed})")
        # Deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Set torch deterministic mode")
    except ImportError:
        pass

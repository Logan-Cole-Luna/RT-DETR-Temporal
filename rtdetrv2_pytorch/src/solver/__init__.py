"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver import DetSolver
from .temporal_det_solver import TemporalDetSolver  # New temporal solver


from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
    'uav_detection': DetSolver,  # Standard UAV detection using existing DetSolver
    'uav_temporal_detection': TemporalDetSolver,  # Temporal UAV detection with new solver
}
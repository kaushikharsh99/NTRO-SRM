"""Scientific evaluation for Sentinel-2 super-resolution products.

Reference evaluation and source-consistency evaluation are intentionally kept
separate. Source consistency is useful quality control, but is not a substitute
for a paired high-resolution benchmark.
"""

from ntro_srm.evaluation.evaluator import EvaluationReport, evaluate_geotiffs
from ntro_srm.evaluation.metrics import evaluate_arrays

__all__ = ["EvaluationReport", "evaluate_arrays", "evaluate_geotiffs"]

from .fid_evaluator import FIDEvaluator
from .is_evaluator import ISEvaluator


EVALUATOR = {
    'fid': FIDEvaluator,
    'inception_score': ISEvaluator,
}

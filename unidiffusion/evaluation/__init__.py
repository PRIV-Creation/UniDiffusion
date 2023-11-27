from .fid_evaluator import FIDEvaluator
from .is_evaluator import ISEvaluator
from .clip_evaluator import CLIPScoreEvaluator


EVALUATOR = {
    'fid': FIDEvaluator,
    'inception_score': ISEvaluator,
    'clip_score': CLIPScoreEvaluator,
}

from .base_evaluator import BaseEvaluator
from torchmetrics.multimodal.clip_score import CLIPScore
from unidiffusion.utils.logger import setup_logger


logger = setup_logger(__name__)


class CLIPScoreEvaluator(BaseEvaluator, CLIPScore):
    name = 'CLIP_Score'

    def __init__(self, clip_model=None, **kwargs):
        super().__init__(clip_model)
        self.clip_model = clip_model

    def compute(self):
        result = super().compute()
        return {
            'clip_score': result.item()
        }

    def update(self, calculate_clip_score, image, text, **kwargs):
        if calculate_clip_score:
            super().update(image, text)

    def __repr__(self):
        return f'CLIPScoreEvaluator:\n' \
                f'  CLIP model: {self.clip_model}\n'

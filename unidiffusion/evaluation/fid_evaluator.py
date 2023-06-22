from .base_evaluator import BaseEvaluator
from torchmetrics.image.fid import FrechetInceptionDistance
import random


class FIDEvaluator(BaseEvaluator, FrechetInceptionDistance):
    def __init__(self, real_image_num, reset_real_features=False, **kwargs):
        super().__init__(reset_real_features=reset_real_features, **kwargs)
        self.real_image_num = real_image_num

    def before_train(self, dataset, accelerator):
        # random select different data for each process
        random.seed(accelerator.process_index)
        evaluate_idx = random.choices(
            range(len(dataset)), k=min(self.real_image_num, len(dataset)) // accelerator.num_processes
        )

        for i in evaluate_idx:
            self.update((dataset[i]["pixel_values"].unsqueeze(0).to(accelerator.device) + 1) / 2, real=True)

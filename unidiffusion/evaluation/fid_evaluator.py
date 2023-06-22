from .base_evaluator import BaseEvaluator
from torchmetrics.image.fid import FrechetInceptionDistance
import random
from tqdm import tqdm
from unidiffusion.utils.logger import setup_logger


logger = setup_logger(__name__)


class FIDEvaluator(BaseEvaluator, FrechetInceptionDistance):
    name = 'FID'

    def __init__(self, real_image_num, reset_real_features=False, normalize=True, **kwargs):
        super().__init__(reset_real_features=reset_real_features, normalize=normalize, **kwargs)
        self.real_image_num = real_image_num

    def before_train(self, dataset, accelerator):
        # random select different data for each process
        random.seed(0)
        total_idx = random.sample(range(len(dataset)), min(self.real_image_num, len(dataset)))
        image_per_process = len(total_idx) // accelerator.num_processes
        process_idx = total_idx[accelerator.process_index * image_per_process: (accelerator.process_index + 1) * image_per_process]

        logger.info(f'[FID] Calculating {len(total_idx)} real image statistics ... ')

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(process_idx)), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for i in process_idx:
            self.update((dataset[i]["pixel_values"].unsqueeze(0).to(accelerator.device) + 1) / 2, real=True)
            progress_bar.update(1)

    def compute(self):
        result = super().compute()
        return {
            'fid': result.item()
        }

    def __repr__(self):
        return f'FIDEvaluator:\n' \
                f'  real_image_num: {self.real_image_num}\n' \
                f'  real_features_num_samples: {self.real_features_num_samples.item()}\n' \
                f'  fake_features_num_samples: {self.fake_features_num_samples.item()}\n'

from .base_evaluator import BaseEvaluator
from torchmetrics.image.inception import InceptionScore
import random
from tqdm import tqdm
from unidiffusion.utils.logger import setup_logger


logger = setup_logger(__name__)


class ISEvaluator(BaseEvaluator, InceptionScore):
    name = 'InceptionScore'

    def __init__(self, real_image_num=0, normalize=True, **kwargs):
        super().__init__(normalize=normalize, **kwargs)
        self.real_image_num = real_image_num

    def before_train(self, dataset, accelerator):
        if self.real_image_num == 0:
            return None

        # random select different data for each process
        random.seed(0)
        total_idx = random.sample(range(len(dataset)), min(self.real_image_num, len(dataset)))
        print(f'{total_idx}, accelerator.num_processes: {accelerator.num_processes}')
        image_per_process = len(total_idx) // accelerator.num_processes
        process_idx = total_idx[accelerator.process_index * image_per_process: (accelerator.process_index + 1) * image_per_process]

        logger.info(f'[FID] Calculating {len(total_idx)} real image statistics ... ')

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(process_idx)), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for i in process_idx:
            self.update((dataset[i]["pixel_values"].unsqueeze(0).to(accelerator.device) + 1) / 2, real=True)
            progress_bar.update(1)

    def update(self, image, real):
        super().update(image)

    def compute(self):
        result = super().compute()
        return {
            'inception-score_kl_mean': result[0].item(),
            'inception-score_kl_std': result[0].item()
        }

    def __repr__(self):
        return f'InceptionScore:\n' \
                f'  real_image_num: {self.real_image_num}\n'

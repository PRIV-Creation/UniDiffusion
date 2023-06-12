from diffusers.optimization import get_scheduler
from diffusion_trainer.config import LazyCall as L


lr_scheduler = L(get_scheduler)(
        name='get_scheduler',
        num_warmup_steps='${..train.lr_warmup_iter}' * '${..train.gradient_accumulation_iter}',
        num_training_steps='${..train.max_iter}' * '${..train.gradient_accumulation_iter}',
)
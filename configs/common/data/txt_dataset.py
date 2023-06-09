from diffusion_trainer.utils.optim import Optimizer
from diffusion_trainer.config import LazyCall as L
from diffusion_trainer.utils.optim import get_default_optimizer_params


optimizer = L(Optimizer)(
    optimizer='SGD',
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0
    ),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
)
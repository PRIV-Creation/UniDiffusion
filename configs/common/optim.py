from diffusion_trainer.utils.optim import get_optimizer
from diffusion_trainer.config import LazyCall as L


optimizer = L(get_optimizer)(
    optimizer='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
)
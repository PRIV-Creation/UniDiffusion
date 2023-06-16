from unidiffusion.utils.optim import get_optimizer
from unidiffusion.config import LazyCall as L


optimizer = L(get_optimizer)(
    optimizer='AdamW',
    lr=1e-4,
    weight_decay=1e-2,
)
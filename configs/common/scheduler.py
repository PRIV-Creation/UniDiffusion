from diffusers.optimization import get_scheduler
from unidiffusion.config import LazyCall as L


lr_scheduler = L(get_scheduler)(
        name='constant',
)

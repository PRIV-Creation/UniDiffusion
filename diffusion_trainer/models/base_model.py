from diffusion_trainer.utils.logger import setup_logger


class BaseModel:
    trainable = False
    params_train_args = dict()

    @classmethod
    def from_pretrained(cls, training_args=None, *args, **kwargs):
        cls.trainable = training_args is not None
        cls.params_train_args = training_args
        setup_logger(__name__).info('Model {} trainable: {}.'.format(cls.__name__, cls.trainable))
        return super().from_pretrained(*args, **kwargs)

    def get_trainable_params(self):
        if self.trainable:
            return self.parameters()
        else:
            return None



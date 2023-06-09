import torch


class BaseModel(torch.nn.Module):
    training = False,
    params_args = dict()

    def get_params(self):
        pass

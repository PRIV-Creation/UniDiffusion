from unidiffusion.peft.proxy import ProxyLayer


class FinetuneLayer(ProxyLayer):
    pass


def set_finetune_layer(model_name, module, name, train_args, proxy_model):
    module.requires_grad_(True)
    module.original_name = name
    proxy_model.params_group.append(dict(params=module.parameters(), **train_args['optim_kwargs']))
    getattr(proxy_model, model_name).append(module)

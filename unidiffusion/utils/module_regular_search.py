import re


PREDEFINED_PATTERN_UNET = {
    'attention': r'attn(0|1)',
    'cross_attention': r'attn2',
    'cross_attention.q': r'attn2\.to_q',
    'cross_attention.k': r'attn2\.to_k',
    'cross_attention.v': r'attn2\.to_v',
    'cross_attention.qkv': r'attn2\.(to_q|to_k|to_v)',
    'feedforward': r'ff',
    'resnets': r'resnet',
    'resnets.conv': r'resnets\.\d\.conv'
}


def get_module_pattern(input_module, pattern):
    if pattern in PREDEFINED_PATTERN_UNET:
        pattern = PREDEFINED_PATTERN_UNET[pattern]
    last_match_name = "LAST MATCHED MODULE'S NAME"
    for name, module in input_module.named_modules():
        if bool(re.search(pattern, name)):
            if name.startswith(last_match_name):
                continue
            last_match_name = name
            yield module, name


def get_module_type(input_module, module_type):
    last_match_name = "LAST MATCHED MODULE'S NAME"
    for name, module in input_module.named_modules():
        if isinstance(module, module_type):
            if name.startswith(last_match_name):
                continue
            last_match_name = name
            yield module, name

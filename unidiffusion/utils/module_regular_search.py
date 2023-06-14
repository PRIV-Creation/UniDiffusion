import re


def get_module_pattern(input_module, pattern):
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

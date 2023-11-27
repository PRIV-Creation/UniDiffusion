import torch


def collate_fn(examples):
    batch_data = dict()
    for key in examples[0].keys():
        if key == "pixel_values":
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            batch_data[key] = pixel_values
        elif key == "input_ids":
            input_ids = torch.stack([example["input_ids"] for example in examples])
            if input_ids.dim() == 3 and input_ids.shape[1] == 1:
                input_ids = input_ids.squeeze(1)
            batch_data[key] = input_ids
        elif isinstance(examples[0][key], torch.Tensor):
            batch_data[key] = torch.stack([example[key] for example in examples])
        else:
            batch_data[key] = [example[key] for example in examples]
    return batch_data


import torch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    if input_ids.dim() == 3 and input_ids.shape[1] == 1:
        input_ids = input_ids.squeeze(1)
    prompts = [example["prompt"] for example in examples]
    return {"pixel_values": pixel_values, "input_ids": input_ids, "prompt": prompts}

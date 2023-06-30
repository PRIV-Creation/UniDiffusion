#!/bin/bash

# Set the path to the directory containing the .safetensors and .ckpt files
dir_path="/nas/workspace/webui_models/models/Stable-diffusion/civitai/"

# Loop through all .safetensors files in the directory and convert them to diffusers format
for file in "$dir_path"/*.safetensors; do
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    dump_path="/nas/workspace/diffusers_models/$filename"
    if [ ! -d "$dump_path" ]; then
        mkdir "$dump_path"
        python scripts/tools/convert_to_diffusers.py \
            --checkpoint_path "$file" \
            --original_config_file v1-inference.yaml \
            --dump_path "$dump_path" \
            --to_safetensors \
            --from_safetensors
    fi
done

# Loop through all .ckpt files in the directory and convert them to diffusers format
for file in "$dir_path"/*.ckpt; do
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    dump_path="/nas/workspace/diffusers_models/$filename"
    if [ ! -d "$dump_path" ]; then
        mkdir "$dump_path"
        python scripts/tools/convert_to_diffusers.py \
            --checkpoint_path "$file" \
            --original_config_file v1-inference.yaml \
            --dump_path "$dump_path" \
            --to_safetensors
    fi
done
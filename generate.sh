#!/bin/bash
cmd="target/release/nano_gpt generate"
prompts="21+22, 22+21 99+11, 99-10, 90-12, 1+1+1, 10+1+2"
seq_len=15
save_dir=save

if [[ $# -eq 1 ]]; then
  model_params="-m $save_dir/$1.safetensors"
else
  echo "Error: Please provide the epoch to load from"
  exit
fi

IFS=', ' read -r -a array <<< "$prompts"
for element in "${array[@]}"
do
  prompt_params="$prompt_params -p $element="
done


$cmd $model_params $prompt_params --greedy -n $seq_len

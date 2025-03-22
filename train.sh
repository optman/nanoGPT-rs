#!/bin/bash
cmd="target/release/nano_gpt"
sft_input="input-sft.txt"
rl_input="input-rl.txt"
test_input="input-test.txt"
prompts="21+22, 22+21 99+11, 99-10, 90-12, 1+1+1, 10+1+2"
seq_len=15
batch_size=100
save_dir=save
#epoch_max=10
epoch_save=100
rollout_temperature=3.0
rollout_num=16
clip_ratio=0
pi_iters=2


sft_iter=10
rl_iter=10
max_loop=100

# Example:./train.sh 
# Example:./train.sh 100 

if [[ $# -eq 1 ]]; then
  last_epoch=$1
else
  last_epoch=0
fi

#check if $last_epoch is 0
if [[ $last_epoch -eq 0 ]]; then
  model_params=""
else
  model_params="-m $save_dir/${last_epoch}.safetensors"
fi

IFS=', ' read -r -a array <<< "$prompts"
for element in "${array[@]}"
do
  prompt_params="$prompt_params -p $element="
done


pretrain_cmd=pretrain
sft_cmd=sft
rl_cmd="rl --clip-ratio $clip_ratio --rollout-temperature $rollout_temperature  --rollout-num $rollout_num --pi-iters $pi_iters"



# Function to build the training command
build_cmd() {
  local epoch_max=$1
  local cmd_type=$2
  local input=$3
  echo "$cmd train --input $input $prompt_params $model_params --seq-len $seq_len --batch-size $batch_size --epoch-max $epoch_max --epoch-save $epoch_save $cmd_type"
}

# Function to update the last epoch
update_last_epoch() {
  local epoch_increment=$1
  last_epoch=$((last_epoch + epoch_increment))
  model_params="-m $save_dir/${last_epoch}.safetensors"

  echo "***************** Eval ************************************************"
  $cmd eval --input $test_input $model_params -n $seq_len --batch-size $batch_size
  echo "***********************************************************************"
}

# Main loop
for i in $(seq 1 $max_loop)
do
  # Self-finetuning (SFT)
  echo "---------- SFT training ($i/$max_loop) --------------------------------"
  sft_train=$(build_cmd "$sft_iter" "$sft_cmd" "$sft_input")
  echo $sft_train
  $sft_train
  update_last_epoch $sft_iter


  # Reinforcement learning (RL)
  echo "========== RL training ($i/$max_loop) ================================="
  rl_train=$(build_cmd "$rl_iter" "$rl_cmd" "$rl_input")
  echo $rl_train
  $rl_train
  update_last_epoch $rl_iter

done








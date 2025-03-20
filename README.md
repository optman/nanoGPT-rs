# Training a LLM for Basic Arithmetic

This project demonstrates training a tiny Language Model (LLM) to perform basic arithmetic operations using Supervised Fine-tuning (SFT) and Reinforcement Learning (RL).

## Dataset

The training data consists of simple arithmetic operations:

```
1+1=2
11+12=23
1+1+1=2+1=3
11-10=1
...
```

## Training Approach

1. **Supervised Fine-tuning (SFT)**: Initial training on the dataset.
2. **Reinforcement Learning (RL)**: Using GRPO (Group Relative Policy Optimization) algorithm.
   - Prompts are rolled out multiple times.
   - Rewards: 1.0 for correct answers, -1.0 for incorrect ones.
   - Advantage calculation: (reward - mean) / standard deviation.

We alternate between SFT and RL to accelerate learning.

## Model Details

- Character-level encoding
- Vocabulary size: 128
- Training samples: 2,000
- RotateRotary positional encoding

## Training Process

Run `./train.sh` to start the training. Here's a summary of the training progression:

1. **Initial Output**: Random characters
2. **Early RL**: Mostly incorrect predictions (reward ≈ -1)
3. **Mid-training (1000 epochs)**: ~80% accuracy, some operations correct
4. **Later stages**: Occasional correct answers for harder operations (e.g., 99+11)
5. **Final stage (2000 epochs)**: 
   - Near 100% accuracy on training data
   - Still struggles with some operations (e.g., 99+11, 21+22)


## Training Logs and Observations

Below is a detailed breakdown of the training logs, showcasing the model's progression through supervised fine-tuning (SFT) and reinforcement learning (RL). These logs highlight the challenges and milestones encountered during training.

You can run the training process using the command `./train.sh`. Below is a summary of the results observed during training:

1. **Initial Training**: The model starts with randomly initialized parameters and quickly learns to produce outputs after a few epochs.
    ```
    ---------- SFT Training (1/100) ----------
    Epoch 1: Average loss = 0.00327, Elapsed time = 2s
    Predictions: 
    21+22=tnK[[, 22+21=X X X, 22-21=tnK[[, 1+1+1==tnoa, 99+11=m53k6, 99-11=vUvUvU, 10+1+2=X X X
    ...
    Epoch 4: Average loss = 0.00233, Elapsed time = 2s
    Predictions: 
    21+22=1, 22+21=1, 22-21=1, 1+1+1=1, 99+11=1, 99-11=1, 10+1+2=1
    ```

2. **Early Reinforcement Learning (RL)**: Initially, the model's predictions are mostly incorrect, resulting in rewards close to -1.
    ```
    ========== RL Training (1/100) ==========
    Epoch 11: Average reward = -0.98262, Average loss = 0.00449, Elapsed time = 65s
    Predictions: 
    21+22=1, 22+21=1, 22-21=1, 1+1+1=1, 99+11=1, 99-11=1, 10+1+2=1
    ...
    Epoch 13: Average reward = -0.98845, Average loss = -0.00053, Elapsed time = 50s
    Predictions: 
    21+22=1, 22+21=1, 22-21=1, 1+1+1=1, 99+11=1, 99-11=1, 10+1+2=1
    ```

3. **Mid-Training Progress (1000 Epochs)**: The model achieves approximately 80% accuracy, correctly predicting some operations like `99-11` and `1+1+1`.
    ```
    ---------- SFT Training (60/100) ----------
    Epoch 1181: Average loss = 0.00004, Elapsed time = 2s
    Predictions: 
    21+22=42, 22+21=41, 22-21=1, 1+1+1=2+1=3, 99+11=119, 99-11=88, 10+1+2=3=3
    ...
    ========== RL Training (60/100) ==========
    Epoch 1191: Average reward = 0.82131, Average loss = -0.00174, Elapsed time = 51s
    Predictions: 
    21+22=4, 22+21=4, 22-21=1, 1+1+1=2+1=3, 99+11=119, 99-11=88, 10+1+2=3=3
    ```

4. **Occasional Correct Predictions**: At certain points, the model correctly predicts harder operations like `99+11`, but it struggles to retain this ability consistently.
    ```
    Epoch 1395: Average reward = 0.96595, Average loss = -0.00034, Elapsed time = 50s
    Predictions: 
    21+22=4, 22+21=4, 22-21=1, 1+1+1=2+1=3, 99+11=110, 99-11=88, 10+1+2=2=3
    ```

5. **Final Stage (2000 Epochs)**: By the end of training, the model achieves near 100% accuracy on the training data but still struggles with certain operations like `99+11` and `21+22`.
    ```
    Epoch 2000: Average reward = 1.00952, Average loss = 0.00000, Elapsed time = 50s
    Predictions: 
    21+22=4, 22+21=4, 22-21=1, 1+1+1=2+1=3, 99+11=119, 99-11=88, 10+1+2=3=3
    ```

These results highlight the model's ability to memorize training data while struggling to generalize to more complex or unseen arithmetic operations.

## Conclusions

The LLM shows limited ability to generalize arithmetic rules beyond the training data. I have written a blog [post](https://blog.optman.net/can-llms-really-learn-arithmetic/) on this experiment.
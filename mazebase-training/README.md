# Ask Your Humans: Using Human Instructions to Improve Generalization in Reinforcement Learning

This section of the repository contains code for training using the dataset.

## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

The input requires the dataset to be split into the state, action, inventory, and goal. You can download the whole dataset and easily do this.

## IL Training

To train the model(s) in the paper, run this command:

```train
python3 train_<method>.py
```
For IL only, use train_bc.py
For our method, use train_hierarchy.py
For SP, use train_stateprediction.py
For SR, use train_autoencoder.py

## RL Training

To train the model(s) in the paper, run this command:

```train
python3 main.py --env-name "mazebase-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --log-dir <file_dir_to_save>--num-env-steps 100000000 --model-name <my_model_name>
```

These are the hyperparameters used for training all models which receive RL reward. Code was mainly borrowed and modified from [here](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).

## Evaluation

To evaluate a trained model, run:

```eval
python3 enjoy.py --load-dir <model_path>
```

## Pre-trained Models

Some of the pretrained models are uploaded [here](https://bit.ly/2GXROwf).
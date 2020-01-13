# TRPO Tensorflow 2
TRPO Implementation for Reinforcement Learning Project @ Sapienza

This project was done as for a Reinforcement Learning Class in the Master's Degree in Artificial Intelligence and Robotics taught by prof. Roberto Capobianco.

## Requirements
Create a conda environment with the following command
`$ conda create -n trpo python`

Activate the environment just created
`$ conda activate trpo`

Install the requirements with the following command
`$ pip install -r requirements.txt`

## Training
To train on an environment you have to create a file MyEnv.py inside the folder configs. The config file must declare 4 variables:
	- config: A dictionary with the parameters for the TRPO Agent
	- env: the gym environment
	- policy_model: a Tensorflow 2.0 Keras Sequential model for the Policy
	- value_model: a Tensorflow 2.0 Keras Sequential model for the Value

To run the training run the following command:
`$ python train.py MyEnv`

For example to train a MountainCar-v0 agent
`$ python train.py MountainCar-v0`

The program will prompt out a log dir that can be opened with tensorboard. The same log dir will contain the policy model checkpoints.

PS: do not add .py to MyEnv while running the command above

## Testing
To test on an environment you to run the following command:

To run the training run the following command:
`$ python test.py path/to/saved/model/weight.ckpt MyEnv`

For example to run MountainCar-v0 pretrained agent
`$ python test.py saved_models/MountainCar-v0/490.ckpt MountainCar-v0`

PS: the file .ckpt does't have to exist. Do not add .py to MyEnv while running the command above

## Saved Models
Some pre-trained models are available in the folder saved_models.

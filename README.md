# Udacity Reinforcement Learning Continuous Control:
### Using Deep Deterministic Policy Gradient:

This repository describe the solution for the continuous control project in the Udacity Deep Reinforcement Learning Nanodegree.

## Environment Description:

![Agent](./model_test.gif)  
  
There are 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm:
We used Reacher_2 which has 20 agents to solve the environment.

Set-up: Double-jointed arm which can move to target locations.
Goal: The agents must move its hand to the goal location, and keep it there.
Agent Reward Function (independent):
+0.1 Each step agent's hand is in goal location.
Benchmark Mean Reward: 30


## Repository Description:
Reacher_1.app and Reacher_2.app are the environment
model.py: contains the actor and critic NN which is composed of fully connected NN with 2 hidden layers (256, 128).
Saved_Model/checkpoint_actor.pth & Saved_Model/checkpoint_critic.pth: contains the saved Weights
Continuous Control.ipynb: jupyter notebook environment to train the environment
Report.md: Contains the analysis report.
run.py: to test the trained network
train.py: contains the agent description.

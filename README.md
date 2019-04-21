# RL_Continuous_Control
Using Proximal Based Methods to solve RL problems

 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

 Set-up: Double-jointed arm which can move to target locations.
Goal: The agents must move its hand to the goal location, and keep it there.
Agents: The environment contains 10 agent linked to a single Brain.
Agent Reward Function (independent):
+0.1 Each step agent's hand is in goal location.
Brains: One Brain with the following observation/action space.
Vector Observation space: 26 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
Visual Observations: None.
Reset Parameters: Two, corresponding to goal size, and goal movement speed.
Benchmark Mean Reward: 30

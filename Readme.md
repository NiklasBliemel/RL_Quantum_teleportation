# Training an Reinforcement Learning Agent to perform Quantum Teleportation

In this project, we train a Reinforcement Learning (RL) agent to solve
the quantum teleportation problem with 3 Q bits. We use a MLP network and the Maskable PPO algorithm 
from the RL library Stable Baselines 3.

## To Run the experiment:

Install all requirements
```
pip install -r requirements.txt 
```
Train agents (be aware you will override the previous test results):
```
python main.py
```
Plot the results:
```
python plot.py
```

## Further Analysis

In labfuncs.py, you can find test functions to further analyse your agents. It is suggested to use them in jupyter lab
(jupyter is installed with requirements.txt).

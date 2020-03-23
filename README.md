# Self Supervised Imitation Learning with Synthetic Experts

## Install

First, install gym and atari environments. You may need to install other dependencies depending on your system.

```
pip install gym
```

and then install atari with one of the following commands
```
pip install "gym[atari]"
pip install gym[atari]
```

We also require you to use a version greater than 1 for Tensorflow.


## Environment

### Pong-v0

- We play against a decent AI player.
- One player wins if the ball pass through the other player and gets reward +1 else -1.
- Episode is over when one of the player reaches 21 wins
- final score is between -21 or +21 (lost all or won all)

```python
# action = int in [0, 6)
# state  = (210, 160, 3) array
# reward = 0 during the game, 1 if we win, -1 else
```

We use a modified env where the dimension of the input is reduced to

```python
# state = (80, 80, 1)
```

with downsampling and greyscale.

## Training

First, train an assortment of teacher agents using any method/repo of your choice. Then, to create a series of demonstrations using those trained agents use the env_create_demos.py script for each guide agent. This script will save out the intermediate states visited by the guide and the reward/environment state that can be later restored. For each expert that you want to run for that guide, run env_expert_demos.py on the npz record generated by env_create_demos. Repeat this for all the guide/expert configurations you want to try.

Once you've generated the npz files for all guides/expert, merge them into a single npz file and copy the state_images into the student/ folder. The code for training the student is in student/train.py. 


# RL_toolkits
It contains a source of proximal policy optimization (PPO) as one of the deep RL approaches along with a disturbance mechanism. The idea is straightforward: just a temporal stop (with a certain probability) while the agent explores the environment so that the tutor helps to reduce or correct searching space.


## Objectives
I want to figure out answers to the following questions such that

1. A PPO agent can 'perceive' the target while it learns grasping skills.
2. The disturbance mechanism may reduce or make shortcuts to reach the optimal behaviors.

Here, 'perception' indicates weights in the global average pooling (GAP) layer of an actor and a critic. Or, it can be indirectly checked via tracking behaviors. In general, deep RL in a continuous environment demands a large amount of iterations (~ 1M) due to credit assignment. At least, the agent should notice capturing and keeping the camera on the target is valuable.

## Experiments
Setup, reward signal design, hyperparameters, and results are well explained in the report. Please find it in the journal dir.


## Conclusion
The idea was not successful: the disturbance could not reduce the exploration dramatically. Learning from demonstration (LfD) or curriculum learning should be considered (even in the simulation environment). If a large amount of data is accessible, the agent is able to recognize the target objects.    

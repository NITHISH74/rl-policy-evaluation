# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

#### States
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.
Five Transition states / Non-terminal States including S: The starting state.
#### Actions
The agent can take two actions:

R: Move right.
L: Move left.
#### Transition Probabilities
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

#### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

#### Graphical Representation
![image](https://github.com/HariniBaskar/rl-policy-evaluation/assets/93427253/c99c60ab-9487-4106-b791-876a2ad03d47)


## POLICY EVALUATION FUNCTION
![image](https://github.com/HariniBaskar/rl-policy-evaluation/assets/93427253/6653401e-d973-4bd8-8c31-96cbea332d8c)

## PROGRAM:
```
Developed by: Nithishwar S
Register Number: 212221230071
```
```py
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
# code  to evaluate the given policy
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma+prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
      return V

# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

# Comparing the two policies
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```

## OUTPUT:
### POLICY 1
![2A](https://github.com/HariniBaskar/rl-policy-evaluation/assets/93427253/4d61ee31-c0d0-4a73-aa6b-4fee824fa5ed)

![2A1](https://github.com/HariniBaskar/rl-policy-evaluation/assets/93427253/546d34a1-ee4c-4f68-9ea0-7a75f833c6a1)

### POLICY 2
![2B](https://github.com/HariniBaskar/rl-policy-evaluation/assets/93427253/4be3c41e-df0f-40dd-8921-aea5b4039dde)

![2B2](https://github.com/HariniBaskar/rl-policy-evaluation/assets/93427253/3dbbbd64-ddee-45c7-ae7f-3a957f1c8b84)

### COMPARISON:
![2C](https://github.com/HariniBaskar/rl-policy-evaluation/assets/93427253/7c925610-a404-4406-9c56-849523ade126)

### CONCLUSION:
![2D](https://github.com/HariniBaskar/rl-policy-evaluation/assets/93427253/61ce8bb5-8e9e-499b-bd37-c2c1d818e6b8)

## RESULT:
Thus, a Python program is developed to evaluate the given policy.

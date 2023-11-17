import gym
import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from gym import spaces

#env = gym.make('MountainCarContinuous-v0')
env =  gym.make('MountainCar-v0', render_mode = "human")


states =  env.observation_space.shape[0]
actions = env.action_space.n

#print thr number of actions print(actions)

model = Sequential()
model.add(Flatten (input_shape=(1, states)))
model.add(Dense (24, activation="relu"))
model.add(Dense (24, activation="relu"))
model.add(Dense (actions, activation="linear"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory (limit=50088, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)
agent.compile(Adam (lr=0.001), metrics=["mae"])
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=18, visualize=True)
print(np.mean(results.history["episode_reward"]))
env.close()

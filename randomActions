import random
import gym

#env = gym.make('MountainCarContinuous-v0')
env =  gym.make('MountainCar-v0', render_mode = "human")

states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 10
for episode in range(1, episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = random.choice([0,1])
        _, reward, done, _ = env.step(action)
        score += reward
        env.render()
        print(state)
        print("-------------------------")
        print(action)
        print("-------------------------")
    print(f"Episode {episode} , Score: {score} ")

print(states)
env.close()

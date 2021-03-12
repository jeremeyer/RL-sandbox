import numpy as np
import gym

import sac
from discretenets import *


def main():
    env = gym.make('CartPole-v1')

    obs_size = np.prod(env.observation_space.shape)
    act_size = env.action_space.n

    policynet = discretePolicyNet(obs_size, env.action_space.n, 2, 256)
    qnet = discreteQNet(obs_size, act_size, 2, 256)
    agent = sac.DiscreteSACAgent(policynet, qnet, None, 1000)

    for _ in range(100):
        exploreModel(env, agent)
        agent.experience_replay(100, 50)
    print(get_model_fitness(env, agent))


def exploreModel(env, agent):
    agent.eval()
    transitions = []
    done = False
    env_state = env.reset()
    while not done:
        action, probability = agent.get_policy_result(env_state)
        old_state = env_state
        env_state, step_reward, done, _ = env.step(action)
        transitions.append(sac.Transition(old_state, action, env_state, step_reward))

    agent.add_transitions(transitions)
    agent.train()


def get_model_fitness(env, model):
    model.eval()
    env_state = env.reset()
    reward = 0
    done = False
    while not done:
        a, _ = model.get_action(env_state)
        env_state, step_reward, done, _ = env.step(a)
        reward += step_reward
    model.train()
    return reward


if __name__ == '__main__':
    main()

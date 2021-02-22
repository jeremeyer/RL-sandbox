import gym
import torch
import torch.nn as nn
import copy
from collections import namedtuple
import numpy as np


Transition = namedtuple('Transition', 'state action reward sprime is_terminal')
cuda = torch.device('cuda')


# selects action based on probability distribution
def select_action(env, distribution):
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        actions = [i for i in range(env.action_space.n)]
        return np.random.choice(actions, p=distribution.data.numpy())


# runs exploration and returns a continuous array of undiscounted transitions to form trajectories
def explore(env, model, num_steps=500):
    env_state = env.reset()
    done = False
    trajectories = []
    for t in range(num_steps):
        action_weights = model.forward(torch.from_numpy(env_state).float())
        action = select_action(env, action_weights)
        prev_state = env_state
        env_state, reward, done, info = env.step(action)
        trajectories.append(Transition(prev_state, action, reward, env_state, done))
        if done:
            env_state = env.reset()
    return trajectories


def train_model(env, model, epochs, learning_rate=0.003, gamma=0.99):
    reference_model = copy.deepcopy(model)
    reference_model.eval().cpu()
    model.to(cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        run_epoch(env, model, optimizer, gamma)
    model.cpu()


def run_epoch(env, model, optimizer, gamma):
    reference_model = copy.deepcopy(model).cpu().eval()
    trajectories = explore(env, reference_model)
    states = []
    actions = []
    rewards = []
    current_reward = 0
    for i in range(0, len(trajectories)):
        states.append(trajectories[i].state)
        actions.append(trajectories[i].action)
        current_reward = current_reward * gamma + trajectories[i].reward
        rewards.append(current_reward)
        if trajectories[i].is_terminal:
            current_reward = 0
    states_tensor = torch.Tensor(states).to(cuda)
    actions_tensor = torch.Tensor(actions).to(cuda)
    rewards_tensor = torch.FloatTensor(rewards).to(cuda)
    rewards_tensor /= rewards_tensor.max()
    preds_tensor = model(states_tensor).gather(dim=1, index=actions_tensor.long().view(-1, 1)).squeeze()
    loss = -torch.sum(torch.log(preds_tensor) * rewards_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


class ActorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = self.build_net()

    def build_net(self):
        return nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, data):
        return self.net(data)


def get_model_fitness(env, model):
    model.eval()
    env_state = env.reset()
    reward = 0
    done = False
    while not done:
        a = select_action(env, model(torch.from_numpy(env_state).float()))
        env_state, step_reward, done, _ = env.step(a)
        reward += step_reward
    return reward


def main():
    env = gym.make("CartPole-v1")
    model = ActorModel()
    train_model(env, model, 500)
    print(get_model_fitness(env, model))


if __name__ == '__main__':
    main()

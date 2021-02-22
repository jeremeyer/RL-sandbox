import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy
import gym


# device = torch.device('cuda')
device = torch.device('cpu')


class DiscretePolicyNet(nn.Module):
    def __init__(self, in_dim, num_actions, num_layers, hidden_size):
        super(DiscretePolicyNet, self).__init__()
        self.net = self.build_net(in_dim, num_actions, num_layers, hidden_size)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.num_actions = num_actions
        self.optim = None

    def build_net(self, in_dim, num_actions, num_layers, hidden_size):
        layers = []
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.ReLU())
        for i in range(0, num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_actions))
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)

    def get_action(self, state):
        # make into vector
        state = torch.from_numpy(state).float().unsqueeze(0)

        # make Variable to include info for Autograd
        distro = self.forward(Variable(state))

        selected_action = np.random.choice(self.num_actions, p=np.squeeze(distro.detach().numpy()))
        action_prob = distro.squeeze(0)[selected_action]
        return selected_action, action_prob

    def build_optim(self, lr=3e-4):
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)


def explore(env, model, num_steps=500):
    rewards = []
    action_probs = []
    env_state = env.reset()
    done = False
    while not done:
        action, action_prob = model.get_action(env_state)
        env_state, reward, done, info = env.step(action)
        rewards.append(reward)
        action_probs.append(action_prob)
    return rewards, action_probs


def replay(model, rewards, action_probs, gamma=0.99):
    discounted_rewards = []

    # gamma discounts future rewards dumbfuck
    for i in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[i:]:
            Gt = Gt + gamma**pw * r
            pw += 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards).to(device)

    # normalize rewards to discourage the lowest-scoring actions
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    model.train().to(device)
    if model.optim is None:
        model.build_optim()
    action_probs = torch.stack(action_probs).to(device)

    model.optim.zero_grad()
    loss = -torch.sum(torch.log(action_probs) * discounted_rewards)
    loss.backward()
    model.optim.step()

    model.eval().cpu()


def train_model(env, model, num_epochs):
    for i in range(num_epochs):
        rewards, action_probs = explore(env, model)
        replay(model, rewards, action_probs)


def get_model_fitness(env, model):
    model.eval()
    env_state = env.reset()
    reward = 0
    done = False
    while not done:
        a, _ = model.get_action(env_state)
        env_state, step_reward, done, _ = env.step(a)
        reward += step_reward
    return reward


def main():
    env = gym.make("CartPole-v1")
    num_inputs = 1
    for i in env.observation_space.shape:
        num_inputs *= i
    model = DiscretePolicyNet(num_inputs, env.action_space.n, 2, 24)
    train_model(env, model, 1000)
    print(get_model_fitness(env, model))


if __name__ == '__main__':
    main()

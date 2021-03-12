import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def build_relu_network(in_dim, out_dim, num_layers, neurons_per_layer, end_softmax=False):
    layers = []
    layers.append(nn.Linear(in_dim, neurons_per_layer))
    layers.append(nn.ReLU())
    for i in range(num_layers):
        layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(neurons_per_layer, out_dim))
    if end_softmax:
        layers.append(nn.Softmax(dim=1))
    else:
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class discreteValueNet(nn.Module):
    def __init__(self, in_dim, num_layers, neurons_per_layer):
        super(discreteValueNet, self).__init__()
        self.valuenet = build_relu_network(in_dim, 1, num_layers, neurons_per_layer, end_softmax=False)

    def forward(self, data):
        return self.valuenet.forward(Variable(data))

    def evaluate_state(self, state):
        return self.valuenet.forward(state).squeeze(0)


class discreteQNet(nn.Module):
    def __init__(self, in_dim, num_actions, num_layers, neurons_per_layer):
        super(discreteQNet, self).__init__()
        self.qnet = build_relu_network(in_dim, num_actions, num_layers, neurons_per_layer, end_softmax=False)

    def forward(self, state):
        return self.qnet.forward(Variable(state))

    def evaluate_action_batch(self, states, actions):
        outputs = self.forward(states).squeeze(1)
        return outputs[torch.arange(outputs.size(0)), actions]

    def evaluate_state_tensor(self, state):
        return self.forward(state)

    def evaluate_action_tensor(self, state, action):
        return self.evaluate_state_tensor(state).squeeze()[action]

    def evaluate_state(self, state):
        return self.evaluate_state_tensor(torch.from_numpy(state).float().unsqueeze(0))

    def evaluate_action(self, state, action):
        return self.evaluate_state(state).squeeze(0)[action]


class discretePolicyNet(nn.Module):
    def __init__(self, in_dim, num_actions, num_layers, neurons_per_layer):
        super(discretePolicyNet, self).__init__()
        self.num_actions = num_actions
        self.policynet = build_relu_network(in_dim, num_actions, num_layers, neurons_per_layer, end_softmax=True)

    def forward(self, data):
        return self.policynet.forward(Variable(data))

    # evaluates tensor
    def evaluate_transition_tensor(self, state, action):
        return self.forward(state).squeeze(0)[action]

    # evaluates transition of a state given by a numpy array and a number indicating the action chosen
    # allows the logprob to be calculated/backproped on with an off-policy transition
    def evaluate_transition(self, state, action):
        return self.evaluate_transition_tensor(torch.from_numpy(state).float().unsqueeze(0), action)

    def evaluate_transition_batch(self, states, actions):
        output = self.forward(states).squeeze(1)
        return output[torch.arange(output.size(0)), actions]

    def get_action_from_tensor(self, state):
        probs = self.forward(state)
        action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))

        return action, probs.squeeze(0)[action]

    def get_action(self, state):
        return self.get_action_from_tensor(torch.from_numpy(state).float().unsqueeze(0))

    def get_log_prob(self, state, action):
        return torch.log(self.evaluate_transition(state, action))

    def get_log_prob_batch(self, states, actions):
        return torch.log(self.evaluate_transition_batch(states, actions))

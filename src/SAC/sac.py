import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import collections


Transition = collections.namedtuple("Transition", ['s', 'a', 'sprime', 'r'])


class DiscreteSACAgent(nn.Module):
    def __init__(self, policynet, qnet, valuenet, buffer_size, lr=3e-4):
        super(DiscreteSACAgent, self).__init__()
        self.policynet = policynet
        self.qnet1 = qnet
        self.qnet2 = copy.deepcopy(qnet)
        self.valuenet = valuenet
        self.replay_buffer = []
        self.buffer_size = buffer_size

        self.alpha = 0
        self.log_alpha = torch.zeros(1, requires_grad=True)

        self.q1_optim = torch.optim.Adam(self.qnet1.parameters(), lr=lr)
        self.q2_optim = torch.optim.Adam(self.qnet2.parameters(), lr=lr)
        self.policy_optim = torch.optim.Adam(self.policynet.parameters(), lr=lr)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    def add_transitions(self, transition_arr):
        self.replay_buffer.extend(transition_arr)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = random.sample(self.replay_buffer, self.buffer_size)

    def experience_replay(self, num_epochs, minibatch_size, gamma=0.99):
        for _ in range(num_epochs):
            current_transitions = []
            if minibatch_size < len(self.replay_buffer):
                current_transitions = random.sample(self.replay_buffer, minibatch_size)
            else:
                current_transitions = self.replay_buffer[:]
            rewards = []
            state_tensor = []
            action_tensor = []
            value_estimates = []
            for t in current_transitions:
                rewards.append(t.r)
                state_tensor.append(torch.from_numpy(t.s).float().unsqueeze(0))
                action_tensor.append(t.a)

            state_tensor = torch.stack(state_tensor)
            action_tensor = torch.LongTensor(action_tensor)

            # TODO: need Gval calc

            # loss function building and backprop
            reward_tensor = torch.FloatTensor(rewards)
            q1_tensor = self.qnet1.evaluate_action_batch(state_tensor, action_tensor)
            q2_tensor = self.qnet2.evaluate_action_batch(state_tensor, action_tensor)
            logprob_tensor = self.policynet.get_log_prob_batch(state_tensor, action_tensor)
            # value_tensor = torch.stack(value_estimates)

            # peepolaughing
            q1_loss = F.mse_loss(q1_tensor, reward_tensor)
            q2_loss = F.mse_loss(q2_tensor, reward_tensor)
            value_loss = 0
            policy_loss = (self.alpha * torch.log(logprob_tensor) - torch.min(q1_tensor.clone().detach(), q2_tensor.clone().detach())).mean()

            self.q1_optim.zero_grad()
            self.q2_optim.zero_grad()
            q1_loss.sum().backward()
            q2_loss.sum().backward()
            self.q1_optim.step()
            self.q2_optim.step()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # alpha loss attempt
            alpha_loss = (self.log_alpha * -logprob_tensor.clone().detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = torch.exp(self.log_alpha)

    def get_qval(self, state, action):
        return min(self.qnet1.evaluate_action(state, action), self.qnet2.evaluate_action(state, action))

    def get_state_value(self, state):
        return self.valuenet.evaluate_state(state)

    def get_policy_result(self, state):
        return self.policynet.get_action(state)

    def get_log_prob(self, state, action):
        return self.policynet.get_log_prob(state, action)

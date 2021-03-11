import copy
import random
import torch
import torch.nn.functional as F
import gym
import numpy as np


class DiscreteSACAgent:
    def __init__(self, policynet, qnet, valuenet, buffer_size):
        self.policynet = policynet
        self.qnet1 = qnet
        self.qnet2 = copy.deepcopy(qnet)
        self.valuenet = valuenet
        self.replay_buffer = []
        self.buffer_size = buffer_size

        self.alpha = 0

    def add_transitions(self, transition_arr):
        self.replay_buffer.extend(transition_arr)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = random.sample(self.replay_buffer, self.buffer_size)

    def experience_replay(self, num_epochs, minibatch_size, gamma=0.99):
        for _ in num_epochs:
            current_transitions = random.sample(self.replay_buffer, minibatch_size)
            rewards = []
            qnet1_vals = []
            qnet2_vals = []
            log_probs = []
            value_estimates = []
            for t in current_transitions:
                rewards.append(t.r)
                qnet1_vals.append(self.qnet1.evaluate_state(t.s, t.a))
                qnet2_vals.append(self.qnet2.evaluate_state(t.s, t.a))
                log_probs.append(self.get_log_prob(t.s, t.a))

            # loss function building and backprop
            reward_tensor = torch.stack(rewards)
            q1_tensor = torch.stack(qnet1_vals)
            q2_tensor = torch.stack(qnet2_vals)
            logprob_tensor = torch.stack(log_probs)
            value_tensor = torch.stack(value_estimates)

            # peepolaughing
            q1_loss = F.mse_loss(q1_tensor, reward_tensor)
            q2_loss = F.mse_loss(q2_tensor, reward_tensor)
            value_loss = 0
            policy_loss = (self.alpha * torch.log(logprob_tensor) - torch.min(q1_tensor, q2_tensor)).mean()

    def get_qval(self, state, action):
        return min(self.qnet1.evaluate_action(state, action), self.qnet2.evaluate_action(state, action))

    def get_state_value(self, state):
        return self.valuenet.evaluate_state(state)

    def get_policy_result(self, state):
        return self.policynet.evaluate_state(state)

    def get_log_prob(self, state, action):
        return self.policynet.get_log_prob(state, action)

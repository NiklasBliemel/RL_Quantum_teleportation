import torch
import random
import numpy as np
from time import sleep
import gymnasium as gym
from gymnasium import spaces
from QbitSim.Gate import Gate
from QbitSim.GateList import GateList
from QbitSim.utils import new_q_bits, infidality


class QbitEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    tol = 1e-9
    bell_state = [Gate("H", [1]), Gate("CX", [1, 0])]

    def __init__(self, N_qbits, allowed_gates, max_steps, action_repetition=1, max_cnots=2,
                 starting_aid=False, single_rule=None, double_rule=None, measure_rule=None):
        super(QbitEnv, self).__init__()

        self.actions = GateList(N_qbits, allowed_gates, single_rule, double_rule, measure_rule)
        self.action_counter = torch.zeros(len(self.actions))
        self.action_repetition = action_repetition
        self.action_mask = torch.zeros(len(self.actions), dtype=torch.float64)
        self.action_space = spaces.Discrete(len(self.actions))

        self.state = -torch.ones(max_steps, dtype=torch.int32)
        low = torch.full_like(self.state, -1)
        high = torch.full_like(self.state, 3 * len(self.actions))
        self.observation_space = spaces.Box(low=low.numpy(), high=high.numpy(), dtype=np.int32)

        self.N_qbits = N_qbits
        self.starting_aid = starting_aid
        self.cur_step = 0
        self.max_steps = max_steps
        self.cnot_counter = 0
        self.max_cnots = max_cnots

        self.psi, self.random_state = new_q_bits(L=3)
        if starting_aid:
            for gate in QbitEnv.bell_state:
                self.psi = gate(self.psi)[0]

    def reset(self, seed=None, options=None):
        if options is not None:
            torch.random.manual_seed(seed)

        self.psi, self.random_state = new_q_bits(L=self.N_qbits)
        if self.starting_aid:
            for gate in QbitEnv.bell_state:
                self.psi = gate(self.psi)[0]

        self.state[:] = -1
        self.cur_step = 0
        self.cnot_counter = 0
        self.action_counter[:] = 0.
        self.action_mask[:] = 0.
        return self.state.numpy(), {}

    def step(self, action):
        info = {"Step": self.cur_step}

        if self.cur_step + 1 == self.max_steps:
            info["M"] = None
            return self.state, -10, True, False, info

        if str(self.actions[action])[:2] == "CX":
            if not self.out_of_cnots():
                self.cnot_counter += 1
                if self.out_of_cnots():
                    self.action_mask[self.get_cnot_index()] = -torch.inf

        if self.action_counter[action] < self.action_repetition:
            self.action_counter[action] += 1
            if self.action_counter[action] >= self.action_repetition:
                self.action_mask[action] = -torch.inf

        self.psi, m = self.actions[action](self.psi)

        terminal = bool(self._terminal_check())
        info["M"] = int(m) if m is not None else None
        info["Action"] = str(self.actions[action])
        info["Terminal"] = terminal

        m = 2 if m is None else m
        self.state[self.cur_step] = m * len(self.actions) + action

        self.cur_step += 1
        return self.state.numpy(), -1, terminal, False, info

    def render(self, mode='human'):
        for action in self.state:
            if action != -1:
                gate = str(self.actions[action % len(self.actions)])
                m = action // len(self.actions)
                print(f"Gate: {gate}, M = {m}")

    def get_action_mask(self):
        return self.action_mask

    def _terminal_check(self, inspect=False):
        infi = infidality(self.psi, self.random_state)
        if inspect:
            print(f"Infidelity: {infi:.5e}")
        return infi < QbitEnv.tol

    def test_net(self, qnet):
        try:
            obs, info = self.reset()
            goal_reached = False
            for i in range(self.max_steps):
                q_values = qnet(self.state)
                A = torch.argmax(q_values + self.action_mask)
                obs, R, goal_reached, truncated, info = self.step(A)
                if str(self.actions[A])[0] == "M" and len(self.actions[A].target) == 1:
                    print(f"{self.actions[A]}\tm = {info["M"]}")
                else:
                    print(self.actions[A])
                sleep(0.2)
                if goal_reached:
                    print(f"\nGoal in {i + 1} steps!")
                    break
            if not goal_reached:
                print(f"\nFailed in {i + 1} steps!")
        except KeyboardInterrupt:
            clear_output()

    def qval_test(self, qnet, steps=1):
        S, info = self.reset()
        for i in range(steps):
            q_vals = qnet(self.state)
            A = torch.argmax(q_vals + self.action_mask)
            print(f"\nQ-vals {i}:")
            for j, q_val in enumerate(q_vals + self.action_mask):
                gate = str(self.actions[j])
                if j != A:
                    print(f"{gate}:\t\t {q_val}") if len(gate) < 10 else print(f"{gate}:\t {q_val}")
                else:
                    print(f"{gate}:\t\t {q_val} <---") if len(gate) < 10 else print(f"{gate}:\t {q_val} <---")
            S, R, terminal, trunc, info = self.step(A)
            if terminal:
                print(f"\nGoal in {i + 1} steps!")
                break
        S, info = self.reset()

    def print_actions(self):
        print(self.actions)

    def get_cnot_index(self):
        out = []
        for i, a in enumerate(self.actions):
            if str(a)[:2] == "CX":
                out.append(i)
        return out

    def out_of_cnots(self):
        return self.cnot_counter >= self.max_cnots

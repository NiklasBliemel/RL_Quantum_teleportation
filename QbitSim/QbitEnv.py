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
    tol = 1e-9
    full_solution = [Gate("H", [1]), Gate("CX", [1, 0]),
                     Gate("CX", [2, 1]), Gate("H", [2]),
                     Gate("MX", [1, 0]), Gate("MZ", [2, 0])]

    def __init__(self, N_qbits, allowed_gates, max_steps, starting_aid=2):
        super(QbitEnv, self).__init__()

        self.actions = GateList(N_qbits, allowed_gates)
        self.max_steps = max_steps
        self.start = QbitEnv.full_solution[:starting_aid]
        self.solution = QbitEnv.full_solution[starting_aid:]
        self.shape = (max_steps, 3 * len(self.actions))
        self.cur_step = 0
        self.current_state, self.random_state = new_q_bits(L=3)
        self.feature_vector = torch.zeros(self.shape, dtype=torch.float64).flatten()
        for gate in self.start:
            self.current_state = gate(self.current_state)[0]

        self.action_space = spaces.Discrete(len(self.actions))
        low = torch.full_like(self.feature_vector, 0.)  # Min value for each element
        high = torch.full_like(self.feature_vector, 1.)  # Max value for each element
        self.observation_space = spaces.Box(low=low.numpy(), high=high.numpy(), dtype=np.float64)

    def _terminal_check(self, inspect=False):
        infi = infidality(self.current_state, self.random_state)
        if inspect:
            print(f"Infidelity: {infi:.5e}")
        return infi < QbitEnv.tol

    def step(self, action):
        info = {"Step": self.cur_step}
        if self.cur_step + 1 == self.max_steps:
            info["M"] = None
            return self.feature_vector, -1, bool(self._terminal_check()), True, info
        self.current_state, m = self.actions[action](self.current_state)
        info["M"] = m
        info["Action"] = str(self.actions[action])
        if m is None:
            m = 2
        self.feature_vector = self.feature_vector.reshape(self.shape)
        self.feature_vector[self.cur_step, (m * len(self.actions) + action)] = 1
        self.feature_vector = self.feature_vector.flatten()
        self.cur_step += 1
        return self.feature_vector.numpy(), -1, bool(self._terminal_check()), False, info

    def reset(self, seed=None, options=None):
        if options is not None:
            torch.random.manual_seed(seed)
        self.current_state, self.random_state = new_q_bits(L=3)
        self.feature_vector = torch.zeros(self.shape).flatten()
        for gate in self.start:
            self.current_state = gate(self.current_state)[0]
        self.cur_step = 0
        return self.feature_vector.numpy(), {"Step": self.cur_step, "M": None}

    def render(self):
        out = ""
        S = self.feature_vector.reshape(self.shape)
        S = S.argmax(axis=-1)
        t = S // len(self.actions)
        m = S % len(self.actions)
        m[m == 2] = None
        for i in range(0, self.cur_step):
            out += f"{i}th Action: ({str(self.actions[t[i]])} \t measure: {m[i]})\n"
        print(out)

    def test_solution(self, inspect=False):
        obs, info = self.reset()
        for gate in self.solution:
            self.current_state, m = gate(self.current_state)
            print(f"Gate: {gate}, m = {m}") if inspect else None
        print("Solution True!") if self._terminal_check(inspect) else print("Solution False!")
        obs, info = self.reset()

    def test_net(self, qnet):
        try:
            obs, info = self.reset()
            goal_reached = False
            for i in range(self.max_steps):
                q_values = qnet(self.feature_vector)
                if torch.abs(torch.std(q_values)) < QbitEnv.tol:
                    A = random.randint(0, len(self.actions) - 1)
                else:
                    A = torch.argmax(q_values)
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

    def get_feature_length(self):
        return len(self.feature_vector)

    def get_action_length(self):
        return len(self.actions)

    def print_actions(self):
        print(self.actions)

import torch
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from QbitSim.Gate import Gate
from QbitSim.GateList import GateList
from QbitSim.utils import new_q_bits, infidality


class QbitEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    tol = 1e-9

    def __init__(self, N_qbits, max_steps, max_cnots=2, fail_reward=-1):
        super(QbitEnv, self).__init__()

        self.N_qbits = N_qbits
        self.max_steps = max_steps
        self.max_cnots = max_cnots
        self.fail_reward = fail_reward

        self.actions = GateList(N_qbits)
        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(low=-1, high=3 * len(self.actions), shape=[max_steps], dtype=np.int32)
        self.state = -torch.ones(max_steps, dtype=torch.int32)
        self._new_psi()

        self.action_counter = torch.zeros(len(self.actions))

        self.cnot_counter = 0
        self.cnot_indices = self._get_cnot_indices()

        self.cur_step = 0
        self.max_steps = max_steps

        self.possible_actions = torch.arange(len(self.actions))
        self.invalid_actions: list[int] = []

    def reset(self, seed=None, options=None):
        self._new_psi()
        self.state[:] = -1
        self.cur_step = 0
        self.cnot_counter = 0
        self.action_counter[:] = 0.
        self.invalid_actions: list[int] = []
        return self.state.flatten().numpy(), {}

    def step(self, action):
        info = {"Step": self.cur_step}

        if self.cur_step + 1 == self.max_steps:
            info["M"] = None
            info["Action"] = "Step limit reached"
            info["Terminal"] = True
            return self.state.flatten().numpy(), self.fail_reward, False, True, info

        self._update_invalid_actions(action)

        self.psi, m = self.actions[action](self.psi)

        terminal = bool(self._terminal_check())
        info["M"] = int(m) if m is not None else None
        info["Action"] = str(self.actions[action])
        info["Terminal"] = terminal

        self._update_state(action, m)

        return self.state.flatten().numpy(), -1, terminal, False, info

    def render(self, mode='human'):
        for action in self.state:
            if action != -1:
                gate = str(self.actions[action % len(self.actions)])
                m = action // len(self.actions)
                print(f"Gate: {gate}, M = {m}")

    def action_masks(self) -> list[bool]:
        return [action not in self.invalid_actions for action in self.possible_actions]

    def print_actions(self):
        print(self.actions)

    def _terminal_check(self, inspect=False):
        infi = infidality(self.psi, self.random_state)
        if inspect:
            print(f"Infidelity: {infi:.5e}")
        return infi < QbitEnv.tol

    def _new_psi(self):
        self.psi, self.random_state = new_q_bits(L=self.N_qbits)

    def _update_state(self, action, m):
        m = 2 if m is None else m
        self.state[self.cur_step] = m * len(self.actions) + action
        self.cur_step += 1

    def _update_invalid_actions(self, action):
        if action in self.cnot_indices:
            if self.cnot_counter < self.max_cnots:
                self.cnot_counter += 1
                if self.cnot_counter >= self.max_cnots:
                    self.invalid_actions += self.cnot_indices

        if self.action_counter[action] < 1 and str(self.actions[action])[0] == "M":
            self.invalid_actions.append(action)

    def _get_cnot_indices(self):
        out = []
        for i, a in enumerate(self.actions):
            if str(a)[:2] == "CX":
                out.append(i)
        return out

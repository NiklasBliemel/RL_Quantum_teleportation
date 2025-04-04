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

    def test_net(self, qnet):
        try:
            obs, info = self.reset()
            goal_reached = False
            for i in range(self.max_steps):
                q_values = qnet(self.state)
                A = torch.argmax(q_values + 1e8 * (torch.tensor(self.action_masks()).int() - 1))
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
            A = torch.argmax(q_vals + 1e8 * (torch.tensor(self.action_masks()).int() - 1))
            print(f"\nQ-vals {i}:")
            for j, q_val in enumerate(q_vals + 1e8 * (torch.tensor(self.action_masks()).int() - 1)):
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

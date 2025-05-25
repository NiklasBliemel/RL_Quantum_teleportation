import torch
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from QbitSim.Gate import Gate
from QbitSim.GateList import GateList
from QbitSim.utils import new_q_bits, infidality

'''''
Main part of the project:
QbitEnv defines simulates a Quantum Computer with N qbits.
It also defines all rules for the training process e.g.:
    - staring conditions
    - goal state
    - rewards
    - form of observations
    - max number of actions before reset

It was built to work as Environment for MaskablePPO algorithm, which has the ability to
rule out available actions. E.g. this is used to limit how many CX gates are allowed to
be used in one run.
https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html# for further details.
'''''


class QbitEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    tol = 1e-9

    def __init__(self, N_qbits, max_steps, max_cnots=2, fail_reward=-5):
        super(QbitEnv, self).__init__()

        self.N_qbits = N_qbits
        self.max_steps = max_steps
        self.max_cnots = max_cnots
        self.fail_reward = fail_reward  # defines the reward of max steps is reached without reaching goal state

        self.actions = GateList(N_qbits)
        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(low=-1, high=3 * len(self.actions), shape=[self.max_steps], dtype=np.int32)
        self.state = -torch.ones(self.max_steps, dtype=torch.int32)
        self._new_psi()

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
        self.invalid_actions: list[int] = []
        return self.state.flatten().numpy(), {}

    def step(self, action):
        info = {"Step": self.cur_step}

        if self.cur_step == self.max_steps:
            score = self.case_check()
            info["M"] = None
            info["Action"] = "Step limit reached"
            info["Terminal"] = True
            return self.state.flatten().numpy(), self.fail_reward + score, False, True, info

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

    def eval_circuit(self, circuit) -> int:
        psi, random_state = new_q_bits(L=self.N_qbits)
        for i in range(circuit.shape[0]):
            psi, m = self.actions[circuit[i, 0]](psi, force_m=circuit[i, 1])
            if infidality(psi, random_state) < QbitEnv.tol:
                return 1
        return 0

    def case_check(self) -> int:
        circuit = []
        m_index = []
        score: int = 0
        for i, action in enumerate(self.state):
            a = int(action % len(self.actions))
            m = int(action // len(self.actions))
            if m != 2:
                m = 0
                m_index.append(i)
            circuit.append([a, m])
        circuit = torch.tensor(circuit, dtype=torch.int32)
        for i in range(2 ** len(m_index)):
            for k in range(len(m_index)):
                circuit[m_index[k]][1] = i % 2 ** (k + 1) // 2 ** k
            score += self.eval_circuit(circuit)
        return score

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
        else:
            if str(self.actions[action])[0] == "M":
                self.invalid_actions.append(action)

    def _get_cnot_indices(self):
        out = []
        for i, a in enumerate(self.actions):
            if str(a)[:2] == "CX":
                out.append(i)
        return out

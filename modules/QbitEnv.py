import torch
import random
import numpy as np
from time import sleep
import gymnasium as gym
from gymnasium import spaces
from pyparsing import actions

'''''
Goal is to teleport state from qbit[2] -> qbit[0] 
'''''

dtype = torch.complex128
zero = torch.tensor([1, 0], dtype=dtype)
one = torch.tensor([0, 1], dtype=dtype)

pauliI = torch.eye(2, dtype=dtype)

pauliX = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
pauliY = torch.tensor([[0, 1j], [-1j, 0]], dtype=dtype)
pauliZ = torch.tensor([[1, 0], [0, -1]], dtype=dtype)

hamada = (pauliX + pauliZ) / np.sqrt(2.0)

controlX = torch.zeros(2, 2, 2, 2, dtype=dtype)
controlX[0, :, 0] = pauliI
controlX[1, :, 1] = pauliX

controlZ = torch.zeros(2, 2, 2, 2, dtype=dtype)
controlZ[0, :, 0] = pauliI
controlZ[1, :, 1] = pauliZ


def new_q_bits(L):
    random_state = torch.randn(2, dtype=dtype)
    random_state /= torch.norm(random_state)
    out = torch.zeros([2 for _ in range(L)], dtype=torch.complex128)
    out[0, 0, :] = random_state.clone().detach()
    return out, random_state


def contraction(psi, gate, dims):
    rank = len(psi.shape)
    permute_list = []

    if len(dims) == 1:
        for i in range(rank):
            if i < dims[0]:
                permute_list.append(i + 1)
            elif i == dims[0]:
                permute_list.append(0)
            else:
                permute_list.append(i)
        return torch.tensordot(gate, psi, dims=[[1], dims]).permute(*permute_list)

    elif len(dims) == 2:
        for i in range(rank):
            if i < dims[0] and i < dims[1]:
                permute_list.append(i + 2)
            elif i == dims[0]:
                permute_list.append(0)
            elif i == dims[1]:
                permute_list.append(1)
            elif (dims[0] < i < dims[1]) or (dims[0] > i > dims[1]):
                permute_list.append(i + 1)
            else:
                permute_list.append(i)
        return torch.tensordot(gate, psi, dims=[[2, 3], dims]).permute(*permute_list)

    raise ValueError("contraction indices must be a list of length 1 or 2")


def infidality(psi, rand_state):
    p = torch.tensordot(torch.conj(psi), psi, dims=[[1, 2], [1, 2]])
    q = torch.conj(rand_state).unsqueeze(-1) * rand_state.unsqueeze(0)
    trace = torch.sum(p * q.T)
    return torch.abs(trace - 1)


def measure(psi, dim):
    probability = torch.sum(torch.abs(psi) ** 2, axis=[i for i in range(len(psi.shape)) if i != dim[0]])
    state = random.choices([zero.clone().detach(), one.clone().detach()], probability.tolist())[0]
    out = contraction(psi, torch.outer(state, state), dim)
    out /= torch.norm(out)
    return out, torch.real(state[1]).int()


class Gate:

    gate_dic = {"X": pauliX, "Y": pauliY, "Z": pauliZ,
                "H": hamada, "CX": controlX, "CZ": controlZ}

    def __init__(self, gate, target):

        if gate[0] == "M":
            self.measure = True
        else:
            self.measure = False
        self.gate = gate
        self.target = target

    def __call__(self, psi):

        if not self.measure:
            return contraction(psi, Gate.gate_dic[self.gate], self.target), None
        else:
            out, m = measure(psi, [self.target[0]])
            if len(self.target) == 1:
                return out, m
            elif len(self.target) == 2:
                if m == 1:
                    return contraction(out, Gate.gate_dic[self.gate[-1]], [self.target[1]]), m
                else:
                    return out, m

    def __str__(self):
        if len(self.target) == 1:
            return f"{self.gate}->{str(self.target)}"
        if len(self.target) == 2:
            return f"{self.gate[0]}[{self.target[0]}]->{self.gate[1]}[{self.target[1]}]"


class GateList:
    single_rule = lambda i: True
    double_rule = lambda i,j: abs(i-j) == 1 and 0 not in [i, j]
    measure_rule = lambda i: i != 0

    def __init__(self, N_qbits, allowed_gates="X Y Z H M CX CZ MX MY MZ"):

        singlegate_targets = [[i] for i in range(N_qbits) if GateList.single_rule(i)]
        doublegate_targets = [[i, j] for i in range(N_qbits) for j in range(N_qbits) if GateList.double_rule(i, j)]
        measure_targets = [[i] for i in range(N_qbits) if GateList.measure_rule(i)]

        self.actions = []

        for gate in allowed_gates.split():
            if gate[0] == "M":
                if len(gate) == 1:
                    self.actions += [Gate(gate, target) for target in measure_targets]
                else:
                    self.actions += [Gate(gate, [i, 0]) for i in range(N_qbits) if i != 0]
            elif len(gate) == 1:
                self.actions += [Gate(gate, target) for target in singlegate_targets]
            else:
                self.actions += [Gate(gate, target) for target in doublegate_targets]

        self._check(N_qbits)

    def __getitem__(self, key):
        return self.actions[key]

    def __iter__(self):
        for action in self.actions:
            yield action

    def __str__(self):
        string = f"{len(self)} actions in total\n\n"
        for i, action in enumerate(self.actions):
            if len(str(action)) < 10:
                string += f"{action} \t\t index: {i}\n"
            else:
                string += f"{action} \t index: {i}\n"
        return string

    def __len__(self):
        return len(self.actions)

    def append(self, gate):
        self.actions.append(gate)

    def _check(self, N_qbits):
        psi, random_state = new_q_bits(N_qbits)
        for action in self.actions:
            check = action(psi)


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
                if torch.abs(torch.std(q_values)) < 1e-9:
                    A = random.randint(0, len(self.actions) - 1)
                else:
                    A = torch.argmax(q_values)
                obs, R, goal_reached, truncated, info = self.step(A)
                if str(self.actions[A])[0] == "M" and len(self.actions[A].target) == 1:
                    print(f"{self.actions[A]}")
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

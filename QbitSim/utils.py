import random

import numpy as np
import torch

dtype = torch.complex128
zero = torch.tensor([1, 0], dtype=dtype)
one = torch.tensor([0, 1], dtype=dtype)
pauliI = torch.eye(2, dtype=dtype)
pauliX = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
pauliY = torch.tensor([[0, 1j], [-1j, 0]], dtype=dtype)
pauliZ = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
hamada = (pauliX + pauliZ) / np.sqrt(2.0)
controlX = torch.zeros(2, 2, 2, 2, dtype=dtype)
controlZ = torch.zeros(2, 2, 2, 2, dtype=dtype)
controlX[0, :, 0] = pauliI
controlX[1, :, 1] = pauliX
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

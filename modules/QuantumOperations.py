import torch
import numpy as np


class Gate:
    operand = torch.tensor([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                            [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]],
                           dtype=torch.complex128)
    pauliX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    pauliY = torch.tensor([[0, 1j], [-1j, 0]], dtype=torch.complex128)
    pauliZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    pauliI = torch.eye(2, dtype=torch.complex128)
    hamada = (pauliX + pauliZ) / np.sqrt(2.0)

    def contract(A, B, dim=0):
        rank_B = len(B.shape)
        permute_list = []
        for i in range(rank_B):
            if i < dim:
                permute_list.append(i+1)
            elif i == dim:
                permute_list.append(0)
            else:
                permute_list.append(i)
        return torch.tensordot(A, B, dims=[[1],[dim]]).permute(*permute_list)

    def x(psi, tdim):
        return contract(Gate.pauliX, psi, tdim)

    def y(psi, tdim):
        return contract(Gate.pauliY, psi, tdim)

    def z(psi, tdim):
        return contract(Gate.pauliZ, psi, tdim)

    def hamada(psi, tdim):
        return contract(Gate.hamada, psi, tdim)

    def cx(psi, cdim, tdim):
        rank = len(psi.shape)
        permute_list = []
        for i in range(rank):
            if i < cdim and i < tdim:
                permute_list.append(i + 2)
            elif i == cdim:
                permute_list.append(0)
            elif i == tdim:
                permute_list.append(1)
            elif (cdim < i and i < tdim) or (i < cdim and adim < i):
                permute_list.append(i + 1)
            else:
                permute_list.append(i)
        return torch.tensordot(Gate.operand, psi, dims=[[2, 3], [cdim, tdim]]).permute(*permute_list)

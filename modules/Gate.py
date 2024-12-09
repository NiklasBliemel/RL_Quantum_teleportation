import torch
import numpy as np


class Gate:
    pauliI = torch.eye(2, dtype=torch.complex128)

    pauliX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    pauliY = torch.tensor([[0, 1j], [-1j, 0]], dtype=torch.complex128)
    pauliZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    hamada = (pauliX + pauliZ) / np.sqrt(2.0)

    phase_gate = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex128)
    T_gate = torch.tensor([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=torch.complex128)

    controlX = torch.zeros(2, 2, 2, 2, dtype=torch.complex128)
    controlX[0, :, 0] = pauliI
    controlX[1, 1] = pauliX

    controlZ = torch.zeros(2, 2, 2, 2, dtype=torch.complex128)
    controlZ[0, 0] = pauliI
    controlZ[1, 1] = pauliZ

    swap_gate = torch.zeros(2, 2, 2, 2, dtype=torch.complex128)
    swap_gate[0, 0, 0, 0] = 1
    swap_gate[0, 1, 1, 0] = 1
    swap_gate[1, 0, 0, 1] = 1
    swap_gate[1, 1, 1, 1] = 1

    def contract(A, B, dim=0):
        rank_B = len(B.shape)
        permute_list = []
        for i in range(rank_B):
            if i < dim:
                permute_list.append(i + 1)
            elif i == dim:
                permute_list.append(0)
            else:
                permute_list.append(i)
        return torch.tensordot(A, B, dims=[[1], [dim]]).permute(*permute_list)

    def double_contract(A, B, dim1=0, dim2=1):
        rank = len(psi.shape)
        permute_list = []
        for i in range(rank):
            if i < dim1 and i < dim2:
                permute_list.append(i + 2)
            elif i == dim1:
                permute_list.append(0)
            elif i == dim2:
                permute_list.append(1)
            elif (dim1 < i < dim2) or (dim1 > i > dim2):
                permute_list.append(i + 1)
            else:
                permute_list.append(i)
        return torch.tensordot(A, B, dims=[[2, 3], [dim1, dim2]]).permute(*permute_list)

    def x(psi, tdim):
        return Gate.contract(Gate.pauliX, psi, tdim)

    def y(psi, tdim):
        return Gate.contract(Gate.pauliY, psi, tdim)

    def z(psi, tdim):
        return Gate.contract(Gate.pauliZ, psi, tdim)

    def hamada(psi, tdim):
        return Gate.contract(Gate.hamada, psi, tdim)

    def phase(psi, tdim):
        return Gate.contract(Gate.phase_gate, psi, tdim)

    def T(psi, tdim):
        return Gate.contract(Gate.T_gate, psi, tdim)

    def cx(psi, cdim, tdim):
        return Gate.double_contract(Gate.controlX, psi, cdim, tdim)

    def cz(psi, cdim, tdim):
        return Gate.double_contract(Gate.controlZ, psi, cdim, tdim)

    def swap(psi, tdim):
        return Gate.double_contract(Gate.swap, psi, tdim)
   
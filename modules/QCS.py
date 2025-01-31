import torch
import numpy as np
import random


class Operands:
    pauliI = torch.eye(2, dtype=torch.complex128)

    pauliX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    pauliY = torch.tensor([[0, 1j], [-1j, 0]], dtype=torch.complex128)
    pauliZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    hamada = (pauliX + pauliZ) / np.sqrt(2.0)

    controlX = torch.zeros(2, 2, 2, 2, dtype=torch.complex128)
    controlX[0, :, 0] = pauliI
    controlX[1, :, 1] = pauliX

    controlZ = torch.zeros(2, 2, 2, 2, dtype=torch.complex128)
    controlZ[0, :, 0] = pauliI
    controlZ[1, :, 1] = pauliZ

    gate_dic = {"X" : pauliX, "Y" : pauliY, "Z" : pauliZ,
                "H" : hamada, "CX" : controlX, "CZ" : controlZ}


class Gate:
    def __init__(self, gate, target):

        assert len(target) == len(gate), f"taget indices dont match chosen gate! (gate: {gate} target: {target}"

        if gate[0] == "M":
            self.measure = True
            if len(target) == 2:
                self.gate = Operands.gate_dic[gate[-1]]
            if len(target) == 1:
                self.gate = None
        else:
            self.measure = False
            self.gate = Operands.gate_dic[gate]

        self.string = gate
        self.target = target
        

    def __call__(self, psi):
        if not self.measure:
            return Functions.contraction(psi, self.gate, self.target), 0
            
        mdim = self.target[0]
        axis = []
        for i in range(len(psi.shape)):
            if i != mdim:
                axis.append(i)
        probability = torch.sum(torch.abs(psi) ** 2, axis=axis)
        state = random.choices([Functions.zero, Functions.one], probability.tolist())[0]
        out = Functions.contraction(psi, torch.outer(state,state), [mdim])
        out /= torch.norm(out)
        
        if len(self.target) == 1:
            return out, torch.real(state[1]).int()
            
        if int(state[1]) == 1:
            return Functions.contraction(out, self.gate, [self.target[1]]), 0
        return out, 0
        

    def __str__(self):
        if len(self.target) == 1:
            return f"{self.string}->{str(self.target)}"
        if len(self.target) == 2:
            return f"{self.string[0]}[{self.target[0]}]->{self.string[1]}[{self.target[1]}]"


class Functions:
    zero = torch.tensor([1, 0], dtype=torch.complex128)
    one = torch.tensor([0, 1], dtype=torch.complex128)
    
    def random_state(N_qbits=1):
        out = torch.rand([2 for _ in range(N_qbits)], dtype=torch.complex128)
        return out / torch.norm(out)

    def combine(states):
        assert len(states) > 1, "at least two states are required"
        out = states[0]
        for i in range(1, len(states)):
            out = out.unsqueeze(-1) * states[i].unsqueeze(0)
        return out

    def classic_state(bitlst, as_list=False):
        assert all(bit in {0, 1} for bit in bitlst), "only 1's and 0's allowed"
        states = [Functions.zero if bit == 0 else Functions.one for bit in bitlst]
        if not as_list:
            return Functions.combine(states)
        else:
            return states

    
    def contraction(psi, gate, dims):
        '''
        input:
            psi: wavefunction
            gate: one- or tow-site operator
            conraction_indices: list of indeces to contract of the wavefunction to contract with the operator
        output:
            new wavefunction after contraction
        '''
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
        
                                    
        raise ValueError("conraction_indices must be a list of length 1 or 2")

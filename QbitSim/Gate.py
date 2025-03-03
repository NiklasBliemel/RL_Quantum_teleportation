from QbitSim.utils import pauliX, pauliY, pauliZ, hamada, controlX, controlZ, contraction, measure


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

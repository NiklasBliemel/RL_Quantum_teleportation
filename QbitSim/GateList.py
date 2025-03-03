from QbitSim.utils import new_q_bits
from QbitSim.Gate import Gate


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

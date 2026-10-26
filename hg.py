import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

qc = QuantumCircuit(4, 4)
qc.h(range(4))
qc.barrier()
qc.measure(range(4), range(4))

fig = qc.draw(output="mpl", fold=-1)
fig.savefig("test_circuit.png", dpi=150, bbox_inches="tight")
plt.show()

print("Circuit image saved as test_circuit.png")
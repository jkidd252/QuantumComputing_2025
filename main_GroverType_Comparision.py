# target grovers
from  Grover_Target import grover
from QuantumMethodsClass import QuantumMethods
# sparse target grover
from Grover_Target_sparse import grover_sparse
from QuantumMethodsClass_sparse import QuantumMethods_sparse
# gate grover
from Grover_gate import grover_Gate
#sparse gate grover
from Grover_gate_sparse import grover_Gate_sparse

from matplotlib import pyplot as plt
import numpy as np
import time

def main():
    qubits = []
    t_s = []
    t_norm = []
    t_s_gate = []
    t_norm_gate = []
    for i in range(2, 11):
        qubits.append(i)
        n = i                                # Number of qubits
        target = 1                                           # Searching for |11⟩ (index 3 in binary)
        iterations = int(np.pi / 4 * np.sqrt(2 ** n))        # Optimal iterations
        
        # TARGET SPARSE VERSION
        time1 = time.time() 
        QM = QuantumMethods_sparse()
        first, second = QM.basis_vectors()
        probabilities = grover_sparse(n, target, iterations, first, QM)
        time2 = time.time()
        time_sparse = time2 - time1
        t_s.append(time_sparse)
        print('sparse ' + str(time_sparse))
        
        # TARGET NORMAL VERSION
        time3 = time.time()
        QM = QuantumMethods()
        first, second = QM.basis_vectors()
        probabilities = grover(n, target, iterations, first, QM)
        time4 = time.time()
        time_norm = time4 - time3
        t_norm.append(time_norm)
        print('normal ' + str(time_norm))
        
        # GATE NORMAL VERSION
        time1 = time.time() 
        QM = QuantumMethods()
        first, second = QM.basis_vectors()
        probabilities = grover_Gate(n, target, iterations, first, QM)
        time2 = time.time()
        time_n_gate = time2 - time1
        t_norm_gate.append(time_n_gate)
        print('Gate ' + str(time_sparse))
        
        # TARGET NORMAL VERSION
        time3 = time.time()
        QM = QuantumMethods_sparse()
        first, second = QM.basis_vectors()
        probabilities = grover_Gate_sparse(n, target, iterations, first, QM)
        time4 = time.time()
        time_s_gate = time4 - time3
        t_s_gate.append(time_s_gate)
        print('normal ' + str(time_norm))
        
        # GATE SPARSE VERSION
    
    plt.plot(qubits, t_s, label='Target Sparse')
    plt.plot(qubits, t_norm, label='Target Normal')
    plt.plot(qubits, t_s_gate, label='Gate Sparse')
    plt.plot(qubits, t_norm_gate, label='Gate Normal')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Time [Seconds]')
    plt.legend()
    #plt.yscale('log')
    plt.savefig('time_vs_qubit.png', dpi=1000)
    plt.show()
    
main()
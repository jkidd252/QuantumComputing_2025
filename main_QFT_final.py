import numpy as np
from fractions import Fraction
from tensor_prod import matrix_prod, tensor_prod

def hadamard_gate(n, target):
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I = np.eye(2)
    U = np.array([1])
    for i in range(n):
        U = tensor_prod(U, H if i == target else I)
    return U

def controlled_phase_gate(n, control, target, k):
    N = 2 ** n
    U = np.ones((N,N), dtype=complex)
    for i in range(0, N):
        for j in range(0, N):
            U[i, j] *= np.exp(2j * np.pi *i *j / (2 ** n))
    return U


def qft_matrix(n):
    """Returns the Quantum Fourier Transform (QFT) matrix for n qubits."""
    N = 2 ** n
    QFT = np.zeros((N, N), dtype=complex)

    for j in range(N):
        for k in range(N):
            QFT[j, k] = np.exp(2j * np.pi * j * k / N)

    return QFT / np.sqrt(N)  # Normalize


def iqft_matrix(n):
    """Returns the Inverse Quantum Fourier Transform (IQFT) matrix for n qubits."""
    return np.conjugate(qft_matrix(n)).T  # IQFT is the Hermitian conjugate of QFT


# 受控U门
def controlled_U_gate(total_qubits, control, target, U, exponent):
    U_power = np.linalg.matrix_power(U, exponent)
    N = 2 ** total_qubits
    CU = np.zeros((N, N), dtype=complex)
    for i in range(N):
        bits = list(format(i, '0{}b'.format(total_qubits)))
        if bits[control] == '0':
            CU[i, i] = 1
        else:
            b = int(bits[target])
            for new_bit in [0, 1]:
                bits_new = bits.copy()
                bits_new[target] = str(new_bit)
                j = int("".join(bits_new), 2)
                CU[j, i] += U_power[new_bit, b]
    return CU

def quantum_phase_estimation(t, U, eigenstate):
    total_qubits = t + 1 
    N = 2 ** total_qubits
    state = np.zeros((N, 1), dtype=complex)
    for k in range(2 ** t):
        for b in [0, 1]:
            index = (k << 1) | b
            state[index, 0] = (1 / np.sqrt(2 ** t)) * eigenstate[b]
    for j in range(t):
        exponent = 2 ** (t - j - 1)   
        CU = controlled_U_gate(total_qubits, j, t, U, exponent)
        state = matrix_prod( CU, state )
    IQFT = iqft_matrix(t)
    I_target = np.eye(2, dtype=complex)
    full_operator = np.kron(IQFT, I_target)
    state = matrix_prod( full_operator , state )
    return state

def measure_first_register(state, t):
    total_qubits = t + 1
    probs = {}
    for k in range(2 ** t):
        prob = 0
        for b in [0, 1]:
            index = (k << 1) | b
            prob += np.abs(state[index, 0]) ** 2
        probs[k] = prob
    return probs

def estimate_order(phi, max_denominator=15):
    fraction = Fraction(phi).limit_denominator(max_denominator)
    return fraction.denominator

def order_finding(U, eigenstate, t, max_denominator=15):
    final_state = quantum_phase_estimation(t, U, eigenstate)
    probs = measure_first_register(final_state, t)
    measured_k = max(probs, key=probs.get)
    phi_estimated = measured_k / (2 ** t)
    print("Estimated phase phi:", phi_estimated)
    r = estimate_order(phi_estimated, max_denominator)
    return r

if __name__ == "__main__":
    # phase that we will try to estimate.
    phi = 0.1  
    U = np.array([[1, 0],
                  [0, np.exp(2j * np.pi * phi)]])
    eigenstate = np.array([0, 1])
    
    t = 3 # number of qubits being considered.

    final_state = quantum_phase_estimation(t, U, eigenstate)
    probs = measure_first_register(final_state, t)
    print("The first register measures the probability distribution of the result:")
    for k in sorted(probs.keys()):
        print("State |{:0{width}b}>: {:.3f}".format(k, probs[k], width=t))
    
    r = order_finding(U, eigenstate, t, max_denominator=15)
    print("Estimated order r is:", r)


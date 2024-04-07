import numpy as np
from cvxopt import matrix, solvers


def BPDN(signal, dictionary, lambda_param ):
    n, d = dictionary.shape
    m = 2*d

    # make the numpy arrays needed
    c = np.ones(m) * lambda_param
    q = np.hstack((c, np.zeros(n)))
    P = np.block([[np.zeros((m, m)), np.zeros((m, n))],
                  [np.zeros((n, m)), np.eye(n)]])
    A = np.hstack((dictionary, -dictionary))
    A_tilde = np.hstack((A, np.eye(n)))
    G = np.hstack((-np.eye(m), np.zeros((m, n))))
    h = np.zeros(m)

    # make them to matrices for cvxopt
    q = matrix(q)
    P = matrix(P)
    G = matrix(G)
    h = matrix(h)
    A_tilde = matrix(A_tilde)
    b = matrix(signal)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A=A_tilde, b=b)
    x_tilde = np.array(sol['x']).reshape(-1)

    alpha = x_tilde[:d] - x_tilde[d:2*d]

    return sol, alpha

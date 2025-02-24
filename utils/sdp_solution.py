import numpy as np
import cvxpy as cp
from utils.function_utils import *
from scipy.linalg import sqrtm





def  sdp_solver(params, output_vars=0):
    A, B, C, Q, R, T, X0_hat, W_hat, V_hat, rho, amb_set, tol = (
        params.A,
        params.B,
        params.C,
        params.Q,
        params.R,
        params.T,
        params.X0_hat,
        params.W_hat,
        params.V_hat,
        params.rho,
        params.amb_set,
        params.tol
    )
    n = A.shape[0]
    m = R.shape[0]
    p = V_hat.shape[0]

    #### Creating Block Matrices for SDP ####
    R_block = np.zeros([T, T, m, m])
    C_block = np.zeros([T, T + 1, p, n])
    for t in range(T):
        R_block[t, t] = R[:, :, t]
        C_block[t, t] = C[:, :, t]
    Q_block = np.zeros([n * (T + 1), n * (T + 1)])
    for t in range(T + 1):
        Q_block[t * n : t * n + n, t * n : t * n + n] = Q[:, :, t]

    R_block = np.reshape(R_block.transpose(0, 2, 1, 3), (m * T, m * T))
    C_block = np.reshape(C_block.transpose(0, 2, 1, 3), (p * T, n * (T + 1)))

    # initialize H and G as zero matrices
    G = np.zeros((n * (T + 1), n * (T + 1)))
    H = np.zeros((n * (T + 1), m * T))
    for t in range(T + 1):
        for s in range(t + 1):
            # breakpoint()
            # print(GG[t * n : t * n + n, s * n : s * n + n])
            G[t * n : t * n + n, s * n : s * n + n] = cumulative_product(A, s, t)
            if t != s:
                H[t * n : t * n + n, s * m : s * m + m] = (
                    cumulative_product(A, s + 1, t) @ B[:, :, s]
                )
    D = np.matmul(C_block, G)
    inv_cons = np.linalg.inv(R_block + H.T @ Q_block @ H)

    ### OPTIMIZATION MODEL ###
    E = cp.Variable((m * T, m * T), symmetric=True)
    E_x0 = cp.Variable((n, n), symmetric=True)
    W_var = cp.Variable((n * (T + 1), n * (T + 1)))
    V_var = cp.Variable((p * T, p * T))
    E_w = []
    E_v = []
    W_var_sep = []  # cp.Variable((n*(T+1),n*(T+1)), symmetric=True)
    V_var_sep = []  # cp.Variable((p*T, p*T), symmetric=True)
    for t in range(T):
        if amb_set == "OT":
            E_w.append(cp.Variable((n, n), symmetric=True))
            E_v.append(cp.Variable((p, p), symmetric=True))
        W_var_sep.append(cp.Variable((n, n), symmetric=True))
        V_var_sep.append(cp.Variable((p, p), symmetric=True))
    W_var_sep.append(cp.Variable((n, n), symmetric=True))
    M_var = cp.Variable((m * T, p * T))
    M_var_sep = []
    num_lower_tri = num_lower_triangular_elements(T, T)
    for k in range(num_lower_tri):
        M_var_sep.append(cp.Variable((m, p)))
    k = 0
    cons = []
    cons.append(E >> 0)

    for t in range(T):
        for s in range(t + 1):
            cons.append(M_var[t * m : t * m + m, p * s : p * s + p] == M_var_sep[k])
            cons.append(M_var_sep[k] == np.zeros((m, p)))
            k = k + 1

    for t in range(T + 1):
        cons.append(W_var[n * t : n * t + n, n * t : n * t + n] == W_var_sep[t])
        cons.append(W_var_sep[t] >> 0)

    # Setting the rest of the elements of the matrix to zero
    for i in range(W_var.shape[0]):
        for j in range(W_var.shape[1]):
            # If the element is not in one of the blocks
            if not any(
                n * t <= i < n * (t + 1) and n * t <= j < n * (t + 1)
                for t in range(T + 1)
            ):
                cons.append(W_var[i, j] == 0)

    for t in range(T):
        cons.append(V_var[p * t : p * t + p, p * t : p * t + p] == V_var_sep[t])
        cons.append(V_var_sep[t] >> 0)
        if amb_set == "OT":
            cons.append(E_v[t] >> 0)
            cons.append(E_w[t] >> 0)
    # Setting the rest of the elements of the matrix to zero
    for i in range(V_var.shape[0]):
        for j in range(V_var.shape[1]):
            # If the element is not in one of the blocks
            if not any(
                p * t <= i < p * (t + 1) and p * t <= j < p * (t + 1)
                for t in range(T + 1)
            ):
                cons.append(V_var[i, j] == 0)

    if amb_set == "OT":
        cons.append(E_x0 >> 0)
        cons.append(cp.trace(W_var_sep[0] + X0_hat - 2 * E_x0) <= rho**2)
        cons.append(W_var_sep[0] >> np.min(np.linalg.eigvals(X0_hat)) * np.eye(n))
        for t in range(T):
            cons.append(
                cp.trace(W_var_sep[t + 1] + W_hat[:, :, t] - 2 * E_w[t]) <= rho**2
            )
            cons.append(cp.trace(V_var_sep[t] + V_hat[:, :, t] - 2 * E_v[t]) <= rho**2)

            cons.append(
                W_var_sep[t + 1] >> np.min(np.linalg.eigvals(W_hat[:, :, t])) * np.eye(n)
            )
            cons.append(
                V_var_sep[t] >> np.min(np.linalg.eigvals(V_hat[:, :, t])) * np.eye(p)
            )
        X0_hat_sqrt = sqrtm(X0_hat)
        cons.append(
            cp.bmat(
                [
                    [cp.matmul(cp.matmul(X0_hat_sqrt, W_var_sep[0]), X0_hat_sqrt), E_x0],
                    [E_x0, np.eye(n)],
                ]
            )
            >> 0
        )
        for t in range(T):
            temp = sqrtm(W_hat[:, :, t])
            cons.append(
                cp.bmat(
                    [
                        [cp.matmul(cp.matmul(temp, W_var_sep[t + 1]), temp), E_w[t]],
                        [E_w[t], np.eye(n)],
                    ]
                )
                >> 0
            )
            temp = sqrtm(V_hat[:, :, t])
            cons.append(
                cp.bmat(
                    [
                        [cp.matmul(cp.matmul(temp, V_var_sep[t]), temp), E_v[t]],
                        [E_v[t], np.eye(p)],
                    ]
                )
                >> 0
            )
    else: 
        cons.append( 1/ 2 * (cp.trace(cp.matmul(W_var_sep[0], np.linalg.inv(X0_hat)))- cp.log_det(cp.matmul(W_var_sep[0], np.linalg.inv(X0_hat)))- p)<= rho)
        for t in range(T):
            cons.append( 1/ 2 * (cp.trace(cp.matmul(V_var_sep[t], np.linalg.inv(V_hat[:, :, t])))- cp.log_det(cp.matmul(V_var_sep[t], np.linalg.inv(V_hat[:, :, t])))- p)<= rho)
            cons.append( 1/ 2 * (cp.trace(cp.matmul(W_var_sep[t+1], np.linalg.inv(W_hat[:, :, t])))- cp.log_det(cp.matmul(W_var_sep[t+1], np.linalg.inv(W_hat[:, :, t])))- p)<= rho)



    cons.append(
        cp.bmat(
            [
                [
                    E,
                    cp.matmul(
                        cp.matmul(cp.matmul(cp.matmul(H.T, Q_block), G), W_var), D.T
                    )
                    + M_var / 2,
                ],
                [
                    (
                        cp.matmul(
                            cp.matmul(cp.matmul(cp.matmul(H.T, Q_block), G), W_var),
                            D.T,
                        )
                        + M_var / 2
                    ).T,
                    cp.matmul(cp.matmul(D, W_var), D.T) + V_var,
                ],
            ]
        )
        >> 0
    )
    obj = -cp.trace(cp.matmul(E, inv_cons)) + cp.trace(
        cp.matmul(cp.matmul(cp.matmul(G.T, Q_block), G), W_var)
    )

    prob = cp.Problem(cp.Maximize(obj), cons)
    # breakpoint()
    prob.solve(
        solver="MOSEK",
        mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol},
        verbose=True,
    )
    obj_clean = 0 # This is truly unnecessary

    if output_vars == 1:
        W_hat_arranged = np.zeros([n * (T + 1), n * (T + 1)])
        V_hat_arranged = np.zeros([p * T, p * T])
        W_hat_arranged[0:n, 0:n] = X0_hat
        for t in range(T):
            W_hat_arranged[
                n * (t + 1) : n * (t + 1) + n, n * (t + 1) : n * (t + 1) + n
            ] = W_hat[:, :, t]
        for t in range(T):
            V_hat_arranged[p * t : p * t + p, p * t : p * t + p] = V_hat[:, :, t]

        W_opt_sdp = W_var.value
        V_opt_sdp = V_var.value
        M = M_var.value

        U_opt = (
            -np.linalg.inv(R_block + H.T @ Q_block @ H)
            @ (H.T @ Q_block @ G @ W_opt_sdp @ D.T + M / 2)
            @ np.linalg.inv(D @ W_opt_sdp @ D.T + V_opt_sdp)
        )
        if rho == 0:
            U_opt = (
                -np.linalg.inv(R_block + H.T @ Q_block @ H)
                @ (H.T @ Q_block @ G @ W_hat_arranged @ D.T + M / 2)
                @ np.linalg.inv(D @ W_hat_arranged @ D.T + V_hat_arranged)
            )
            W_opt_sdp = W_hat_arranged
            V_opt_sdp = V_hat_arranged
        breakpoint()
        temp_R = R_block + H.T @ Q_block @ H
        obj_temp = np.trace((D.T @ U_opt.T @ temp_R @ U_opt @ D + G.T @ Q_block @ G  + 2 * G.T @ Q_block @ H @ U_opt @ D) @ W_opt_sdp) + np.trace(U_opt.T @ temp_R @ U_opt @ V_opt_sdp)
    

        return obj.value, U_opt, W_opt_sdp, V_opt_sdp

    return obj.value, obj_clean, W_var.value, V_var.value



def sdp_solver_worst_case(params, U_opt, obj_return_on=0):
    A, B, C, Q, R, T, X0_hat, W_hat, V_hat, rho, amb_set, tol = (
        params.A,
        params.B,
        params.C,
        params.Q,
        params.R,
        params.T,
        params.X0_hat,
        params.W_hat,
        params.V_hat,
        params.rho,
        params.amb_set,
        params.tol,
    )
    n = A.shape[0]
    m = R.shape[0]
    p = V_hat.shape[0]

    #### Creating Block Matrices for SDP ####
    R_block = np.zeros([T, T, m, m])
    C_block = np.zeros([T, T + 1, p, n])
    for t in range(T):
        R_block[t, t] = R[:, :, t]
        C_block[t, t] = C[:, :, t]
    Q_block = np.zeros([n * (T + 1), n * (T + 1)])
    for t in range(T + 1):
        Q_block[t * n : t * n + n, t * n : t * n + n] = Q[:, :, t]

    R_block = np.reshape(R_block.transpose(0, 2, 1, 3), (m * T, m * T))
    C_block = np.reshape(C_block.transpose(0, 2, 1, 3), (p * T, n * (T + 1)))

    # initialize H and G as zero matrices
    G = np.zeros((n * (T + 1), n * (T + 1)))
    H = np.zeros((n * (T + 1), m * T))
    for t in range(T + 1):
        for s in range(t + 1):
            G[t * n : t * n + n, s * n : s * n + n] = cumulative_product(A, s, t)
            if t != s:
                H[t * n : t * n + n, s * m : s * m + m] = (
                    cumulative_product(A, s + 1, t) @ B[:, :, s]
                )
    D = np.matmul(C_block, G)
    temp_R = R_block + H.T @ Q_block @ H

    ### OPTIMIZATION MODEL ###
    # E = cp.Variable((m * T, m * T), symmetric=True)
    W_var = cp.Variable((n * (T + 1), n * (T + 1)))
    V_var = cp.Variable((p * T, p * T))
    E_w = []
    E_v = []
    W_var_sep = []  # cp.Variable((n*(T+1),n*(T+1)), symmetric=True)
    V_var_sep = []  # cp.Variable((p*T, p*T), symmetric=True)
    for t in range(T):
        if amb_set == "OT":
            E_w.append(cp.Variable((n, n), symmetric=True))
            E_v.append(cp.Variable((p, p), symmetric=True))
        W_var_sep.append(cp.Variable((n, n), symmetric=True))
        V_var_sep.append(cp.Variable((p, p), symmetric=True))
    W_var_sep.append(cp.Variable((n, n), symmetric=True))
    cons = []

    for t in range(T + 1):
        cons.append(W_var[n * t : n * t + n, n * t : n * t + n] == W_var_sep[t])
        cons.append(W_var_sep[t] >> 0)
 

    # Setting the rest of the elements of the matrix to zero
    for i in range(W_var.shape[0]):
        for j in range(W_var.shape[1]):
            # If the element is not in one of the blocks
            if not any(
                n * t <= i < n * (t + 1) and n * t <= j < n * (t + 1)
                for t in range(T + 1)
            ):
                cons.append(W_var[i, j] == 0)

    for t in range(T):
        cons.append(V_var[p * t : p * t + p, p * t : p * t + p] == V_var_sep[t])
        cons.append(V_var_sep[t] >> 0)
       
        if amb_set == "OT":
            cons.append(E_v[t] >> 0)
            cons.append(E_w[t] >> 0)
    # Setting the rest of the elements of the matrix to zero
    for i in range(V_var.shape[0]):
        for j in range(V_var.shape[1]):
            # If the element is not in one of the blocks
            if not any(
                p * t <= i < p * (t + 1) and p * t <= j < p * (t + 1)
                for t in range(T + 1)
            ):
                cons.append(V_var[i, j] == 0)

    if amb_set == "OT":
        E_x0 = cp.Variable((n, n), symmetric=True)
        cons.append(E_x0 >> 0)
        cons.append(cp.trace(W_var_sep[0] + X0_hat - 2 * E_x0) <= rho**2)
        cons.append(W_var_sep[0] >> np.min(np.linalg.eigvals(X0_hat)) * np.eye(n))
        for t in range(T):
            cons.append(
                cp.trace(W_var_sep[t + 1] + W_hat[:, :, t] - 2 * E_w[t]) <= rho**2
            )
            cons.append(cp.trace(V_var_sep[t] + V_hat[:, :, t] - 2 * E_v[t]) <= rho**2)

            cons.append(
                W_var_sep[t + 1] >> np.min(np.linalg.eigvals(W_hat[:, :, t])) * np.eye(n)
            )
            cons.append(
                V_var_sep[t] >> np.min(np.linalg.eigvals(V_hat[:, :, t])) * np.eye(p)
            )
        X0_hat_sqrt = sqrtm(X0_hat)
        cons.append(
            cp.bmat(
                [
                    [cp.matmul(cp.matmul(X0_hat_sqrt, W_var_sep[0]), X0_hat_sqrt), E_x0],
                    [E_x0, np.eye(n)],
                ]
            )
            >> 0
        )
        for t in range(T):
            temp = sqrtm(W_hat[:, :, t])
            cons.append(
                cp.bmat(
                    [
                        [cp.matmul(cp.matmul(temp, W_var_sep[t + 1]), temp), E_w[t]],
                        [E_w[t], np.eye(n)],
                    ]
                )
                >> 0
            )
            temp = sqrtm(V_hat[:, :, t])
            cons.append(
                cp.bmat(
                    [
                        [cp.matmul(cp.matmul(temp, V_var_sep[t]), temp), E_v[t]],
                        [E_v[t], np.eye(p)],
                    ]
                )
                >> 0
            )
    else: 
        cons.append( 1/ 2 * (cp.trace(cp.matmul(W_var_sep[0], np.linalg.inv(X0_hat))) - cp.log_det(cp.matmul(W_var_sep[0], np.linalg.inv(X0_hat)))- p)<= rho)
        for t in range(T):
            cons.append( 1/ 2 * (cp.trace(cp.matmul(V_var_sep[t], np.linalg.inv(V_hat[:, :, t])))- cp.log_det(cp.matmul(V_var_sep[t], np.linalg.inv(V_hat[:, :, t])))- p)<= rho)
            cons.append( 1/ 2 * (cp.trace(cp.matmul(W_var_sep[t+1], np.linalg.inv(W_hat[:, :, t])))- cp.log_det(cp.matmul(W_var_sep[t+1], np.linalg.inv(W_hat[:, :, t])))- p)<= rho)


    obj = cp.trace(
        (
            D.T @ U_opt.T @ temp_R @ U_opt @ D
            + G.T @ Q_block @ G
            + 2 * G.T @ Q_block @ H @ U_opt @ D
        )
        @ W_var
    ) + cp.trace((U_opt.T @ temp_R @ U_opt) @ V_var)

    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(
        solver="MOSEK",
        mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol},
        verbose=True,
    )
    W_opt_sdp = W_var.value
    V_opt_sdp = V_var.value
    if obj_return_on == 1:
        return W_opt_sdp, V_opt_sdp, obj.value

    return W_opt_sdp, V_opt_sdp

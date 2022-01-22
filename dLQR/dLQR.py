from __future__ import print_function

import argparse

import control
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import os

def dlyap(A, Q):
    return control.dlyap(A, Q)


def dare(A, B, Q, R):
    return control.dare(A, B, Q, R)


def chol(A):
    return la.cholesky(A, lower=True, check_finite=False)


def ctrb(A, B):
    return control.ctrb(A, B)


def obsv(A, C):
    return control.obsv(A, C)


def rank(A):
    return np.linalg.matrix_rank(A)


def get_P(transition, Q_dare):
    return dlyap(transition.transpose(), Q_dare)


def get_Sigma(transition, X):
    return dlyap(transition, X)


def plot(Cost_list, cost_star,final_cost,max_iteration,num,title):
    figure_size = 3.6, 2.7
    size_line = 1.5
    fig1 = plt.figure(figsize=figure_size)
    ax11 = fig1.add_axes([0.19, 0.155, 0.75, 0.75])
    Cost_list_star = cost_star*np.ones_like(Cost_list[1])
    for i in range(num):
        plt.semilogy((Cost_list[i]-Cost_list_star)/Cost_list[i], linewidth=size_line,label=round(final_cost[i],3))
    plt.legend(title='Final Cost', ncol=2, fontsize=10)
    ax11.set_xlim(0, max_iteration)
    ax11.set_ylim(1e-8, 1)
    plt.tick_params(axis='both', labelsize=10)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Relative Cost Error', fontsize=10)
    plt.title(title, fontsize=10)



class DynamicsModel(object):

    def __init__(self, flag):
        super(DynamicsModel, self).__init__()
        if flag == 1:
            # dimension of state, control, observation
            self.num_state = 1
            self.num_observation = 1
            self.num_control = 1
            # parameters of experiments
            self.num_agent = 2
            # parameters of discrete dynamics
            self.A = np.array([[1.1]], dtype=np.float64)
            self.B = np.array([[1.0]], dtype=np.float64)
            self.C = np.array([[1.0]], dtype=np.float64)
            # weighting matrices of utility function
            self.Q = np.array([[5.0]], dtype=np.float64)
            self.R = np.array([[1.0]], dtype=np.float64)
            self.V = self.R
            # parameters of initial states
            self.X11 = np.array([[1.0]], dtype=np.float64)
            self.X12 = np.array([[0.25]], dtype=np.float64)
            self.X21 = np.array([[0.25]], dtype=np.float64)
            self.X22 = np.array([[1.0]], dtype=np.float64)
            self.X = np.block([[self.X11, self.X12], [self.X21, self.X22]])

        if flag == 2:
            # dimension of state, control, observation
            self.num_state = 1
            self.num_observation = 1
            self.num_control = 1

            # parameters of experiments
            self.num_agent = 2

            # parameters of discrete dynamics
            self.A = np.array([[0.9]], dtype=np.float64)
            self.B = np.array([[1.0]], dtype=np.float64)
            self.C = np.array([[1.0]], dtype=np.float64)

            # weighting matrices of utility function
            self.Q = np.array([[5.0]], dtype=np.float64)
            self.R = np.array([[1.0]], dtype=np.float64)
            self.V = self.R

            # parameters of initial states
            self.X11 = np.array([[1.0]], dtype=np.float64)
            self.X12 = np.array([[0.25]], dtype=np.float64)
            self.X21 = np.array([[0.25]], dtype=np.float64)
            self.X22 = np.array([[1.0]], dtype=np.float64)
            self.X = np.block([[self.X11, self.X12], [self.X21, self.X22]])

        if flag == 3:
            # dimension of state, control, observation
            self.num_state = 2
            self.num_observation = 1
            self.num_control = 1

            # parameters of experiments
            self.num_agent = 4

            # parameters of discrete dynamics
            self.A = np.array([[1.0, 0.05], [0, 1]], dtype=np.float64)
            self.B = np.array([[0], [0.05]], dtype=np.float64)
            self.C = np.array([[1.0, 0]], dtype=np.float64)

            # weighting matrices of utility function
            self.Q = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float64)
            self.R = np.array([[1.0]], dtype=np.float64)
            self.V = self.R

            # parameters of initial states
            self.X11 = np.array([[0.2, 0],
                                 [0, 0.8]], dtype=np.float64)
            self.X12 = np.array([[0.05, 0],
                                 [0, 0.05]], dtype=np.float64)
            self.X21 = np.array([[0.05, 0],
                                 [0, 0.05]], dtype=np.float64)
            self.X22 = np.array([[0.2, 0],
                                 [0, 0.8]], dtype=np.float64)
            self.X = np.block([[self.X11, self.X12], [self.X21, self.X22]])

        # K,L,T,Q_bar,F
        _, _, self.K_star = dare(self.A, self.B, self.Q, self.R)
        _, _, L_transpose = dare(self.A.transpose(), self.C.transpose(),
                                             self.X11 - self.X12 @ la.inv(self.X22) @ self.X21,
                                             0)
        self.L_star = L_transpose.transpose()
        _, _, L_transpose = dare(self.A.transpose(), self.C.transpose(), self.X11, self.V)
        self.L_p2_star = L_transpose.transpose()
        self.Q_bar = np.block([[self.Q, np.zeros((self.num_state, self.num_state))],
                               [np.zeros((self.num_state, self.num_state)),
                                np.zeros((self.num_state, self.num_state))]])
        self.F = np.block([np.zeros((self.num_state, self.num_state)), np.eye(self.num_state)])
        self.T_star = self.X22 @ la.inv(self.X12)

    def judge(self):
        ctrb_matrix = ctrb(self.A, self.B)
        ctrb_rank = rank(ctrb_matrix)
        if ctrb_rank < self.num_state:
            print('(A,B) is not full rank !')
        obsv_matrix = obsv(self.A, self.C)
        obsv_rank = rank(obsv_matrix)
        if obsv_rank < self.num_state:
            print('(A,C) is not full rank !')

    def get_K(self, T, L):
        C_k = -self.K_star @ la.inv(T)
        B_k = T @ L
        A_k = T @ (self.A - self.B @ self.K_star - L @ self.C) @ la.inv(T)
        return C_k, B_k, A_k

    def get_transition_matrix(self, C_k, B_k, A_k):
        transition_matrix = np.block([[self.A, self.B @ C_k], [B_k @ self.C, A_k]])
        a, b = np.linalg.eig(transition_matrix)
        rho = np.max(np.abs(a))
        return transition_matrix, rho

    def get_Q_dare(self, C_k):
        Q_dare = self.Q_bar + self.F.transpose() @ C_k.transpose() @ self.R @ C_k @ self.F
        return Q_dare

    def get_gradient(self, Sigma, C_k, B_k, A_k, P):

        Sigma11, Sigma12, Sigma21, Sigma22 = Sigma[0:self.num_state, 0:self.num_state], Sigma[0:self.num_state,
                                                                                        self.num_state:2 * self.num_state], Sigma[
                                                                                                                            self.num_state:2 * self.num_state,
                                                                                                                            0:self.num_state], Sigma[
                                                                                                                                               self.num_state:2 * self.num_state,
                                                                                                                                               self.num_state:2 * self.num_state]
        P11, P12, P21, P22 = P[0:self.num_state, 0:self.num_state], P[0:self.num_state,
                                                                    self.num_state:2 * self.num_state], P[
                                                                                                        self.num_state:2 * self.num_state,
                                                                                                        0:self.num_state], P[
                                                                                                                           self.num_state:2 * self.num_state,
                                                                                                                           self.num_state:2 * self.num_state]
        gradient_12 = 2 * (
                (self.R + self.B.transpose() @ P11 @ self.B) @ C_k + self.B.transpose() @ P12 @ A_k) @ Sigma22 \
                      + 2 * self.B.transpose() @ (
                              P11 @ self.A + P12 @ B_k @ self.C) @ Sigma12
        gradient_21 = 2 * (P12.transpose() @ self.A + P22 @ B_k @ self.C) @ Sigma11 @ self.C.transpose() \
                      + 2 * (
                              P12.transpose() @ self.B @ C_k + P22 @ A_k) @ Sigma12.transpose() @ self.C.transpose()
        gradient_22 = 2 * (P12.transpose() @ self.B @ C_k + P22 @ A_k) @ Sigma22 \
                      + 2 * (
                              P12.transpose() @ self.A + P22 @ B_k @ self.C) @ Sigma12

        gradient = np.block([[0, gradient_12], [gradient_21, gradient_22]])
        return gradient

    def get_gradient_p2(self, Sigma, C_k, B_k, A_k, P):

        Sigma11, Sigma12, Sigma21, Sigma22 = Sigma[0:self.num_state, 0:self.num_state], Sigma[0:self.num_state,
                                                                                        self.num_state:2 * self.num_state], Sigma[
                                                                                                                            self.num_state:2 * self.num_state,
                                                                                                                            0:self.num_state], Sigma[
                                                                                                                                               self.num_state:2 * self.num_state,
                                                                                                                                               self.num_state:2 * self.num_state]
        P11, P12, P21, P22 = P[0:self.num_state, 0:self.num_state], P[0:self.num_state,
                                                                    self.num_state:2 * self.num_state], P[
                                                                                                        self.num_state:2 * self.num_state,
                                                                                                        0:self.num_state], P[
                                                                                                                           self.num_state:2 * self.num_state,
                                                                                                                           self.num_state:2 * self.num_state]
        gradient_12 = 2 * (
                (self.R + self.B.transpose() @ P11 @ self.B) @ C_k + self.B.transpose() @ P12 @ A_k) @ Sigma22 \
                      + 2 * self.B.transpose() @ (
                              P11 @ self.A + P12 @ B_k @ self.C) @ Sigma12
        gradient_21 = 2 * (P12.transpose() @ self.A + P22 @ B_k @ self.C) @ Sigma11 @ self.C.transpose() \
                      + 2 * (
                              P12.transpose() @ self.B @ C_k + P22 @ A_k) @ Sigma12.transpose() @ self.C.transpose() + 2 * P22 @ B_k @ self.V
        gradient_22 = 2 * (P12.transpose() @ self.B @ C_k + P22 @ A_k) @ Sigma22 \
                      + 2 * (
                              P12.transpose() @ self.A + P22 @ B_k @ self.C) @ Sigma12

        gradient = np.block([[0, gradient_12], [gradient_21, gradient_22]])
        return gradient

    def get_final_P(self, K_bar):
        C_k, B_k, A_k = K_bar[0:self.num_control, self.num_control:self.num_state + self.num_control], K_bar[
                                                                                                       self.num_control:self.num_state + self.num_control,
                                                                                                       0:self.num_observation], K_bar[
                                                                                                                                self.num_control:self.num_control + self.num_state,
                                                                                                                                self.num_observation:self.num_observation + self.num_state]
        transition, rho = self.get_transition_matrix(C_k.copy(), B_k.copy(), A_k.copy())
        Sigma = get_Sigma(transition.copy(), self.X)
        Q_dar = self.get_Q_dare(C_k.copy())
        P = get_P(transition.copy(), Q_dar.copy())
        return P, Sigma, rho

    def get_final_P_p2(self, K_bar):
        C_k, B_k, A_k = K_bar[0:self.num_control, self.num_control:self.num_state + self.num_control], K_bar[
                                                                                                       self.num_control:self.num_state + self.num_control,
                                                                                                       0:self.num_observation], K_bar[
                                                                                                                                self.num_control:self.num_control + self.num_state,
                                                                                                                                self.num_observation:self.num_observation + self.num_state]
        transition, rho = self.get_transition_matrix(C_k.copy(), B_k.copy(), A_k.copy())
        p2_X = np.block([[self.X11, np.zeros([self.num_state, self.num_state])],
                          [np.zeros([self.num_state, self.num_state]), B_k @ self.V @ B_k.transpose()]])
        Sigma = get_Sigma(transition.copy(), p2_X)
        Q_dar = self.get_Q_dare(C_k.copy())
        P = get_P(transition.copy(), Q_dar.copy())
        return P, Sigma, rho

    def policy_gradient(self, C_k_star, B_k_star, A_k_star, K_bar, max_iteration):
        K_list = []
        cost_list = []
        P, Sigma, rho = self.get_final_P(K_bar.copy())
        if rho >= 1:
            print('rho >= 1')
        cost = np.trace(P @ self.X)
        K_list.append(K_bar)
        cost_list.append(cost)
        self.K_bar_star = np.block([[0, C_k_star], [B_k_star, A_k_star]])
        P_star, Sigma_star, rho_star = self.get_final_P(self.K_bar_star.copy())
        if rho_star >= 1:
            print('rho >= 1')
        cost_star = np.trace(P_star @ self.X)
        iter = 0
        while (iter <= max_iteration):
            iter = iter + 1
            if iter % 1000 == 0:
                print('iter = ', iter)
                print('error = ', np.linalg.norm(K_bar - self.K_bar_star, ord='fro'))
                print('grad = ', grad)
            C_k, B_k, A_k = K_bar[0:self.num_control, self.num_control:self.num_state + self.num_control], K_bar[
                                                                                                           self.num_control:self.num_state + self.num_control,
                                                                                                           0:self.num_observation], K_bar[
                                                                                                                                    self.num_control:self.num_control + self.num_state,
                                                                                                                                    self.num_observation:self.num_observation + self.num_state]
            grad = self.get_gradient(Sigma.copy(), C_k.copy(), B_k.copy(), A_k.copy(), P.copy())
            # line search
            theta = 0.01
            beta = 0.5
            step_size = 1
            next_cost = cost
            while (next_cost < 0 or rho >= 1 or (cost - next_cost) < step_size * theta * la.norm(grad,
                                                                                                 ord='fro') ** 2):
                step_size = beta * step_size
                K_bar_next = K_bar - step_size * grad
                P, _, rho = self.get_final_P(K_bar_next)
                next_cost = np.trace(P @ self.X)
            K_bar = K_bar_next
            P, Sigma, rho = self.get_final_P(K_bar.copy())
            cost = np.trace(P @ self.X)
            if iter % 1000 == 0:
                print('cost-cost_star: ')
                print(cost - cost_star)
            cost_list.append(cost)
            K_list.append(K_bar)
        return K_list, cost_list, cost_star, self.K_bar_star

    def policy_gradient_p2(self, C_k_star, B_k_star, A_k_star, K_bar, max_iteration):
        K_list = []
        cost_list = []

        C_k, B_k, A_k = K_bar[0:self.num_control, self.num_control:self.num_state + self.num_control], K_bar[
                                                                                                       self.num_control:self.num_state + self.num_control,
                                                                                                       0:self.num_observation], K_bar[
                                                                                                                                self.num_control:self.num_control + self.num_state,
                                                                                                                                self.num_observation:self.num_observation + self.num_state]
        P, Sigma, rho = self.get_final_P_p2(K_bar.copy())
        if rho >= 1:
            print('rho >= 1')
        p2_X = np.block([[self.X11, np.zeros([self.num_state, self.num_state])],
                          [np.zeros([self.num_state, self.num_state]), B_k @ self.V @ B_k.transpose()]])
        cost = np.trace(P @ p2_X)
        K_list.append(K_bar)
        cost_list.append(cost)
        self.K_bar_star = np.block([[0, C_k_star], [B_k_star, A_k_star]])
        P_star, Sigma_star, rho_star = self.get_final_P_p2(self.K_bar_star.copy())
        if rho_star >= 1:
            print('rho >= 1')
        p2_X = np.block([[self.X11, np.zeros([self.num_state, self.num_state])],
                          [np.zeros([self.num_state, self.num_state]), B_k_star @ self.V @ B_k_star.transpose()]])
        cost_star = np.trace(P_star @ p2_X)
        iter = 0
        while (iter <= max_iteration):
            iter = iter + 1
            if iter % 10 == 0:
                print('iter = ', iter)
                print('error = ', np.linalg.norm(K_bar - self.K_bar_star, ord='fro'))
                print('grad = ', grad)
            C_k, B_k, A_k = K_bar[0:self.num_control, self.num_control:self.num_state + self.num_control], K_bar[
                                                                                                           self.num_control:self.num_state + self.num_control,
                                                                                                           0:self.num_observation], K_bar[
                                                                                                                                    self.num_control:self.num_control + self.num_state,
                                                                                                                                    self.num_observation:self.num_observation + self.num_state]
            grad = self.get_gradient_p2(Sigma.copy(), C_k.copy(), B_k.copy(), A_k.copy(), P.copy())
            # line search
            theta = 0.01
            beta = 0.5
            step_size = 1
            next_cost = cost
            while (next_cost < 0 or rho >= 1 or (cost - next_cost) < step_size * theta * la.norm(grad,
                                                                                                 ord='fro') ** 2):
                step_size = beta * step_size
                K_bar_next = K_bar - step_size * grad
                P, _, rho = self.get_final_P_p2(K_bar_next)
                C_k, B_k, A_k = K_bar_next[0:self.num_control,
                                self.num_control:self.num_state + self.num_control], K_bar_next[
                                                                                     self.num_control:self.num_state + self.num_control,
                                                                                     0:self.num_observation], K_bar_next[
                                                                                                              self.num_control:self.num_control + self.num_state,
                                                                                                              self.num_observation:self.num_observation + self.num_state]
                p2_X = np.block([[self.X11, np.zeros([self.num_state, self.num_state])],
                                  [np.zeros([self.num_state, self.num_state]), B_k @ self.V @ B_k.transpose()]])
                next_cost = np.trace(P @ p2_X)
            K_bar = K_bar_next
            P, Sigma, rho = self.get_final_P_p2(K_bar.copy())
            p2_X = np.block([[self.X11, np.zeros([self.num_state, self.num_state])],
                              [np.zeros([self.num_state, self.num_state]), B_k @ self.V @ B_k.transpose()]])
            cost = np.trace(P @ p2_X)
            if iter % 10 == 0:
                print('cost-cost_star: ')
                print(cost - cost_star)
            cost_list.append(cost)
            K_list.append(K_bar)
        return K_list, cost_list, cost_star, self.K_bar_star


def run(flag, K_bar, num, max_iteratin):
    env = DynamicsModel(flag=flag)
    T_star = env.X22 @ la.inv(env.X12)
    C_k_star, B_k_star, A_k_star = env.get_K(T_star, env.L_star)
    for i in range(num):
        result = env.policy_gradient(C_k_star, B_k_star, A_k_star, K_bar[i], max_iteratin)
        np.save('cost'+str(flag)+'/cost_list-' + str(i) + '.npy', result)


def run_p2(flag, K_bar, num, max_iteration):
    env = DynamicsModel(flag=flag)
    C_k_star, B_k_star, A_k_star = env.get_K(np.eye(env.num_state), env.L_p2_star)
    for i in range(num):
        result = env.policy_gradient_p2(C_k_star, B_k_star, A_k_star, K_bar[i], max_iteration)
        np.save('cost'+str(flag)+'/p2_cost_list-' + str(i) + '.npy', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', default=1, help='1: system 1, 2: system 2, 3: system 3')
    parser.add_argument('--ini_num', type=int, default=4, help='Num of the initial state')
    parser.add_argument('--K_bar_initial', type=float, default=None, help='Initial value of K_bar')
    args = parser.parse_args()
    num = args.ini_num
    system = args.system
    K_bar_initial = args.K_bar_initial
    if K_bar_initial is None:
        if system == 1:
            K_bar_initial = np.zeros([num, 2, 2])
            K_bar_initial[0] = np.array([[0, -1], [1, -1]])
            K_bar_initial[1] = np.array([[0, -0.5], [2, -0.5]])
            K_bar_initial[2] = np.array([[0, 0.8], [-1.8, -0.8]])
            K_bar_initial[3] = np.array([[0, 1.5], [-2, -2]])
            max_iter_p1 = 20000
            max_iter_p2 = 100
        if system == 2:
            K_bar_initial = np.zeros([num, 2, 2])
            K_bar_initial[0] = np.array([[0, -1], [1, -1]])
            K_bar_initial[1] = np.array([[0, -0.5], [2, -0.5]])
            K_bar_initial[2] = np.array([[0, 0.8], [-1.8, -0.8]])
            K_bar_initial[3] = np.array([[0, 1.5], [-0.5, -0.6]])
            max_iter_p1 = 20000
            max_iter_p2 = 100
        if system == 3:
            K_bar_initial = np.zeros([num, 3, 3])
            K_bar_initial[0] = np.array([[0, -2, -3], [0.4, 0.6, 0.06], [0.7, -0.8, 0.9]])
            K_bar_initial[1] = np.array([[0, -2 * 2, -2 * 3], [0.5 * 0.4, 0.6, 0.06], [0.5 * 0.7, -0.8, 0.9]])
            K_bar_initial[2] = np.array([[0, 2 * 2.5, 3 * 2.5], [-1 / 2.3 * 0.4, 0.7, 0.07], [-1 / 2.3 * 0.7, -0.7, 0.8]])
            K_bar_initial[3] = np.array([[0, 1.1 * 2, 1.1 * 3], [-0.4 / 1.1, 0.6, 0.06], [-0.7 / 1.1, -0.8, 0.9]])
            max_iter_p1 = 100000
            max_iter_p2 = 2000
    print('./cost' + str(system) + '/')
    if not os.path.exists('./cost' + str(system) + '/'):
        os.mkdir('./cost' + str(system) + '/')
    run(system, K_bar_initial, num, max_iter_p1)
    run_p2(system, K_bar_initial, num, max_iter_p2)

    cost_p1 = np.ones([num, max_iter_p1+2], dtype="float64")
    cost_p2 = np.ones([num, max_iter_p2+2], dtype="float64")
    final_cost_p1 = []
    final_cost_p2 = []

    if system == 1:
        for i in range(num):
            K_list_p1, cost_list_p1, cost_star_p1, K_bar_star_p1 = np.load('cost1/cost_list-' + str(i) + '.npy', allow_pickle=True)
            cost_p1[i] = cost_list_p1
            final_cost_p1.append(cost_list_p1[-1])


            K_list_p2, cost_list_p2, cost_star_p2, K_bar_star_p2 = np.load('cost1/p2_cost_list-' + str(i) + '.npy', allow_pickle=True)
            cost_p2[i] = cost_list_p2
            final_cost_p2.append(cost_list_p2[-1])
        plot(cost_p1,  cost_star_p1, final_cost_p1, max_iter_p1,num,'Problem 1')
        plot(cost_p2,  cost_star_p2, final_cost_p2, max_iter_p2,num,'Problem 2')
        plt.show()


    if system == 2:
        for i in range(num):
            K_list_p1, cost_list_p1, cost_star_p1, K_bar_star_p1 = np.load('cost2/cost_list-' + str(i) + '.npy', allow_pickle=True)
            cost_p1[i] = cost_list_p1
            final_cost_p1.append(cost_list_p1[-1])


            K_list_p2, cost_list_p2, cost_star_p2, K_bar_star_p2 = np.load('cost2/p2_cost_list-' + str(i) + '.npy',allow_pickle=True)
            cost_p2[i] = cost_list_p2
            final_cost_p2.append(cost_list_p2[-1])
        plot(cost_p1,  cost_star_p1, final_cost_p1, max_iter_p1,num,'Problem 1')
        plot(cost_p2,  cost_star_p2, final_cost_p2, max_iter_p2,num,'Problem 2')
        plt.show()

    if system == 3:
        for i in range(num):
            K_list_p1, cost_list_p1, cost_star_p1, K_bar_star_p1 = np.load('cost3/cost_list-' + str(i) + '.npy', allow_pickle=True)
            cost_p1[i] = cost_list_p1
            final_cost_p1.append(cost_list_p1[-1])


            K_list_p2, cost_list_p2, cost_star_p2, K_bar_star_p2 = np.load('cost3/p2_cost_list-' + str(i) + '.npy', allow_pickle=True)
            cost_p2[i] = cost_list_p2
            final_cost_p2.append(cost_list_p2[-1])
        plot(cost_p1,  cost_star_p1, final_cost_p1, max_iter_p1,num,'Problem 1')
        plot(cost_p2,  cost_star_p2, final_cost_p2, max_iter_p2,num,'Problem 2')
        plt.show()



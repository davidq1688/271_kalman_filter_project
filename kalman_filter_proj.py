import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Statistics and Parameters:
omega = 0.2  # angular frequency for the true acceleration profile
def accel_true(time):
    return 10*np.sin(omega*time)

def true_state_propagate(curr_time, current_state):
    # current_state = current_state.reshape((-1,))
    curr_pos = current_state[0,0]
    curr_vel = current_state[1,0]
    curr_a = accel_true(curr_time)
    pos = curr_pos + curr_vel*dt_imu + curr_a*dt_imu**2/2
    vel = curr_vel + curr_a*dt_imu
    return np.array([[pos], [vel], [current_state[2,0]]])

# IMU
imu_fs = 200  # IMU sample rate (Hz)
dt_imu = 1/imu_fs
W = 0.0004  # IMU meas. noise variance (m/s/s)**2
def generate_imu_noise_w():
    w = np.sqrt(W)*np.random.randn()  # imu process noise
    return w  # scalar w

# initial bias statistics
ba_bar = 0
ba_var = 0.01  # (m/s/s)**2
# initial vel statistics
v0_bar = 100  # (m/s)
v0_var = 1  # (m/s)**2
# initial pos statistics
p0_bar = 0
p0_var = 10

# GPS
V_eta_p = 1  # variance of x meas. noise (m)**2
V_eta_v = (4/100)**2  # variance of v meas. noise (m/s)**2
V_gps = np.array([[V_eta_p, 0],
                  [0, V_eta_v]])
gps_fs = 5
dt_gps = 1/gps_fs
gps_meas_interval = int(imu_fs/gps_fs)

def generate_gps_noise_eta():
    eta_p = np.sqrt(V_eta_p)*np.random.randn()
    eta_v = np.sqrt(V_eta_v)*np.random.randn()
    return np.array([eta_p, eta_v]).reshape((-1,1))  # col vector [eta_p, eta_v]

def gps_meas_update(current_state_true, eta):
    current_state_true = current_state_true.reshape((-1,1))
    eta = eta.reshape((-1,1))
    z = current_state_true[0:2].reshape((-1,1)) + eta
    return z  # col vector [z_p, z_v]

# Discrete Linear System: state x = [delta_p_e, delta_v_e, ba], noise w = [w]
# x_k+1 = PHI*x_k + GAMMA*w
# delta_p_e = p_e - p_c, delta_v_e = v_e - v_c
PHI = np.array([[1, dt_imu, -dt_imu**2/2],
                [0, 1, -dt_imu],
                [0, 0, 1]])  # System State Matrix
GAMMA = np.array([[-dt_imu**2/2],
                  [-dt_imu],
                  [0]])  # System noise Matrix
# [delta_z_p, delta_z_v].T = [delta_p, delta_v].T + [eta_p, eta_v].T
H = np.array([[1, 0, 0],
              [0, 1, 0]])

# Initialize true initial states and the filter initial states
def initialize():
    ba_true = ba_bar + np.sqrt(ba_var)*np.random.randn()
    delta_v0_true = np.sqrt(v0_var)*np.random.randn()
    delta_p0_true = np.sqrt(p0_var)*np.random.randn()
    delta_x0_true = np.array([delta_p0_true, delta_v0_true, ba_true]).reshape((-1,1))

    delta_x0 = np.array([0, 0, 0]).reshape((-1,1))

    M0 = np.array([[p0_var, 0,0],
                   [0, v0_var, 0],
                   [0, 0, ba_var]])
    return delta_x0_true, delta_x0, M0

# Simulate one realization, initialize with random true state and random estimated state
# Output: x_true_list (N_time x 3 ndarray)
# Output: delta_x_est_list (N_time x 3 ndarray)
# Output: P_list (list of N_time 3x3 P matrix)
# Output: state_error_list (N_time x 3 ndarray): delta_x_true_list - delta_x_est_list
# Output: residual_list (N_time_gps x 2 ndarray): delta_z - H@delta_x_pred
def simulate_one_realization(time_list, gps_interval=40):
    # initialize system true state, estimate state, and state covariance
    delta_x0_true, delta_x0_est, M0 = initialize()
    
    # state [dp_e = pe - pc, dv_e = ve - vc, ba]
    delta_x_true_list = []
    delta_x_est_list = []
    P_list = []
    x_true_list = []
    residual_list = []  # residual = delta_z - H@delta_x_pred

    delta_x_true_list.append(delta_x0_true)
    # delta_x_est_list.append(delta_x0_est)
    # P_list.append(P0)
    x_true_list.append(delta_x0_true + np.array([[p0_bar],[v0_bar],[0]]))
    M = M0
    delta_x_pred = delta_x0_est

    for k in range(len(time_list)):
        # Propagate True state for t_k+1
        x_true_list.append(true_state_propagate(time_list[k], x_true_list[k]))

        # Aposteriori Estimation at t_k
        if k % gps_interval == 0:
            # measurement obtained
            eta = generate_gps_noise_eta()
            delta_z = gps_meas_update(delta_x_true_list[k], eta)
            # KF meas update: x_est = x_pred + K*(z - H*x_pred), K = P*H.T*inv(V)
            P = M - M@H.T@np.linalg.inv(H@M@H.T + V_gps)@H@M
            r = delta_z - H@delta_x_pred
            delta_x_est = delta_x_pred + P@H.T@np.linalg.inv(V_gps)@r
            residual_list.append(r)
        else:
            # no measurement obtained
            P = M
            delta_x_est = delta_x_pred
        
        P_list.append(P)
        delta_x_est_list.append(delta_x_est)
        
        # imu noise
        w = generate_imu_noise_w()
        # true state propagate t_k+1
        delta_x_true = PHI@delta_x_true_list[k] + GAMMA*w
        delta_x_true_list.append(delta_x_true)

        # Apriori prediction for t_k+1
        delta_x_pred = PHI@delta_x_est_list[k]
        M = PHI@P_list[k]@PHI.T + GAMMA@GAMMA.T*W

    delta_x_true_list = np.array(delta_x_true_list[0:-1]).reshape((-1, 3))
    delta_x_est_list = np.array(delta_x_est_list).reshape((-1, 3))
    state_error_list = delta_x_true_list - delta_x_est_list
    residual_list = np.array(residual_list).reshape((-1, 2))
    x_true_list = np.array(x_true_list[0:-1]).reshape((-1, 3))
    return x_true_list, delta_x_est_list, P_list, state_error_list, residual_list

# Simulate N realizations and Analyze ensamble data
# Output: P_avg_list: [N_time] x np.array(3x3). list of P_ave at each time.
# Output: e_avg_list: (3 x N_time)ndarray. avg of e_l.
# Output: ortho_e_and_e_est_list: list [N_time]. avg of np.dot(e_l - e_avg, x_est).
# Output: residual_all_realization: list [N_time]. list of delta_z - H@delta_x_pred.
def simulate_N_realizations(t_list, N_realization):
    # Ensamble of Realizations
    P_list_all_realization = []  # N_realization x N_time x np.array(3 x 3)
    e_l_all_realization = []  # shape: N_realization x np.array(N_time x 3)
    error_p_all_realization = []  # shape: N_realization x N_time
    error_v_all_realization = []
    error_b_all_realization = []
    x_true_all_realization = []  # shape: N_realization x np.array(N_time x 3)
    # delta_x_est_all_realization = []  # shape: N_realization x np.array(N_time x 3)
    residual_all_realization = []  # shape: N_realization x np.array(N_time_gps x 2)

    for i in tqdm(range(N_realization), desc='Simulating Realizations'):
        # simulate one realization
        x_true_list, delta_x_est_list, P_list, error_l, r_l = simulate_one_realization(t_list)

        # save results
        error_p = np.array(error_l[:, 0]).reshape((-1,))
        error_v = np.array(error_l[:, 1]).reshape((-1,))
        error_b = np.array(error_l[:, 2]).reshape((-1,))

        error_p_all_realization.append(error_p)
        error_v_all_realization.append(error_v)
        error_b_all_realization.append(error_b)

        P_list_all_realization.append(P_list)
        e_l_all_realization.append(error_l)
        x_true_all_realization.append(x_true_list)
        # delta_x_est_all_realization.append(delta_x_est_list)
        residual_all_realization.append(r_l)

    error_p_avg = np.mean(error_p_all_realization, axis=0)  # avg of pos_error[time]
    error_v_avg = np.mean(error_v_all_realization, axis=0)  # avg of vel_error[time]
    error_b_avg = np.mean(error_b_all_realization, axis=0)  # avg of bias_error[time]
    e_avg_list = np.array([error_p_avg, error_v_avg, error_b_avg])  # shape: 3xN_time

    # Compute P_ave, check for x_est and e_est orthogonality
    P_avg_list = []  # N_time x np.array(3x3)
    ortho_e_and_e_est_list = []  # [N_time] list
    for i in tqdm(range(len(t_list)), desc='Computing P_avg and Check Orthogonality'):
        error_l_t = [e_l[i, :].reshape((-1,)) for e_l in e_l_all_realization]
        error_l_t = np.array(error_l_t).T  # shape: 3xN_realization
        temp = error_l_t - e_avg_list[:, i].reshape((-1, 1))  # e_l(ti) - e_avg(ti) shape: 3xN_realization
        # delta_x_est_t = [dx_est[i, :].reshape((-1,)) for dx_est in delta_x_est_all_realization]
        # delta_x_est_t = np.array(delta_x_est_t).T  # shape: 3xN_realization
        x_true_t = [x_true[i, :].reshape((-1,)) for x_true in x_true_all_realization]
        x_true_t = np.array(x_true_t).T  # shape: 3xN_realization
        x_est_t = x_true_t - error_l_t
        P_t_list = []
        ortho_t_list = []
        for j in range(N_realization):
            P_t_list.append(np.outer(temp[:, j], temp[:, j]))
            ortho_t_list.append(np.inner(temp[:, j], x_est_t[:, j]))
        P_avg_list.append(np.mean(P_t_list, axis=0))
        ortho_e_and_e_est_list.append(np.mean(ortho_t_list, axis=0))
    
    return P_avg_list, e_avg_list, ortho_e_and_e_est_list, residual_all_realization

# visualization functions
def plot_one_realization_result(t_list, state_error_list, P_list, P_avg_list=None):
    sigma_p_list = []
    sigma_v_list = []
    sigma_ba_list = []
    for i in range(len(t_list)):
        P_i = P_list[i]
        sigma_p_list.append(np.sqrt(P_i[0,0]))
        sigma_v_list.append(np.sqrt(P_i[1,1]))
        sigma_ba_list.append(np.sqrt(P_i[2,2]))

    plt.figure()
    plt.subplot(131)
    plt.plot(t_list, state_error_list[:, 0], label='pos estimation error')
    plt.plot(t_list, sigma_p_list, 'r', label='1-sigma bound')
    plt.plot(t_list, -1*np.array(sigma_p_list), 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.ylim([-1, 1])
    plt.legend()
    plt.title('Position Estimation Error')
    plt.grid()

    plt.subplot(132)
    plt.plot(t_list, state_error_list[:, 1], label='vel estimation error')
    plt.plot(t_list, sigma_v_list, 'r', label='1-sigma bound')
    plt.plot(t_list, -1*np.array(sigma_v_list), 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.ylim([-0.05, 0.05])
    plt.legend()
    plt.title('Velocity Estimation Error')
    plt.grid()

    plt.subplot(133)
    plt.plot(t_list, state_error_list[:, 2], label='bias estimation error')
    plt.plot(t_list, sigma_ba_list, 'r', label='1-sigma bound')
    plt.plot(t_list, -1*np.array(sigma_ba_list), 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('bias (m/s/s)')
    plt.ylim([-0.05, 0.05])
    plt.legend()
    plt.title('IMU bias Estimation Error')
    plt.grid()
    
    if P_avg_list is not None:
        plot_P_error_one_realization(t_list, P_avg_list, P_list)

def plot_P_error_one_realization(t_list, P_avg_list, P_list):
    P_error_list = []
    for i in range(len(t_list)):
        P_error_list.append(P_avg_list[i] - P_list[i])
    P11_error = [P[0,0] for P in P_error_list]  # shape: 1D len(t_list)
    P12_error = [P[0,1] for P in P_error_list]
    P13_error = [P[0,2] for P in P_error_list]
    P21_error = [P[1,0] for P in P_error_list]
    P22_error = [P[1,1] for P in P_error_list]
    P23_error = [P[1,2] for P in P_error_list]
    P31_error = [P[2,0] for P in P_error_list]
    P32_error = [P[2,1] for P in P_error_list]
    P33_error = [P[2,2] for P in P_error_list]
    plt.figure()
    plt.subplot(331)  # P11 error
    plt.plot(t_list, P11_error)
    plt.title('P_11 Error')
    plt.grid()
    plt.subplot(332)
    plt.plot(t_list, P12_error)
    plt.title('P_12 Error')
    plt.grid()
    plt.subplot(333)
    plt.plot(t_list, P13_error)
    plt.title('P_13 Error')
    plt.grid()
    plt.subplot(334)
    plt.plot(t_list, P21_error)
    plt.title('P_21 Error')
    plt.grid()
    plt.subplot(335)
    plt.plot(t_list, P22_error)
    plt.title('P_22 Error')
    plt.grid()
    plt.subplot(336)
    plt.plot(t_list, P23_error)
    plt.title('P_23 Error')
    plt.grid()
    plt.subplot(337)
    plt.plot(t_list, P31_error)
    plt.title('P_31 Error')
    plt.grid()
    plt.subplot(338)
    plt.plot(t_list, P32_error)
    plt.title('P_32 Error')
    plt.grid()
    plt.subplot(339)
    plt.plot(t_list, P33_error)
    plt.title('P_33 Error')
    plt.grid()
    # plt.show()

def plot_avg_estimation_error(t_list, error_avg_list):
    sigma_p_list = []
    sigma_v_list = []
    sigma_ba_list = []
    for i in range(len(t_list)):
        P_i = P_avg_list[i]
        sigma_p_list.append(np.sqrt(P_i[0,0]))
        sigma_v_list.append(np.sqrt(P_i[1,1]))
        sigma_ba_list.append(np.sqrt(P_i[2,2]))
    plt.figure()
    plt.plot(t_list, error_avg_list[0, :], label='Avg pos est error')
    plt.plot(t_list, np.zeros((len(t_list), )), 'r')
    # plt.plot(t_list, sigma_p_list, 'r', label='1-sigma bound')
    # plt.plot(t_list, -1*np.array(sigma_p_list), 'r')
    plt.xlabel('Time (s)')
    plt.title('Average Position Error')
    plt.grid()

    plt.figure()
    plt.plot(t_list, error_avg_list[1, :], label='avg vel est error')
    plt.plot(t_list, np.zeros((len(t_list), )), 'r')
    # plt.plot(t_list, sigma_v_list, 'r', label='1-sigma bound')
    # plt.plot(t_list, -1*np.array(sigma_v_list), 'r')
    plt.xlabel('Time (s)')
    plt.title('Average Velocity Error')
    plt.grid()

    plt.figure()
    plt.plot(t_list, error_avg_list[2, :], label='avg bias est error')
    plt.plot(t_list, np.zeros((len(t_list), )), 'r')
    # plt.plot(t_list, sigma_ba_list, 'r', label='1-sigma bound')
    # plt.plot(t_list, -1*np.array(sigma_ba_list), 'r')
    plt.xlabel('Time (s)')
    plt.title('Average Bias Error')
    plt.grid()

def plot_orthogonality_for_state_est_and_est_error(t_list, ortho_list):
    plt.figure()
    plt.plot(t_list, ortho_list)
    plt.plot(t_list, np.zeros((len(t_list), )), 'r')
    plt.xlabel('Time (s)')
    plt.title('Check Orthogonality for State Estimate and Estimation Error')
    plt.grid()

def print_residual_correlation(t_gps_list, r_l_all_realization):
    random_time_idx = np.random.choice(range(len(t_gps_list)), size=2, replace=False)
    t_i_idx = random_time_idx[0]
    t_j_idx = random_time_idx[1]

    r_l_ti = [r_l[t_i_idx, :].reshape((-1,)) for r_l in r_l_all_realization]
    r_l_tj = [r_l[t_j_idx, :].reshape((-1,)) for r_l in r_l_all_realization]
    r_l_ti = np.array(r_l_ti).T  # shape: 2xN_realization
    r_l_tj = np.array(r_l_tj).T  # shape: 2xN_realization
    r_l_corr_list = []
    for j in range(N_realization):
        r_l_corr_list.append(np.inner(r_l_ti[:, j], r_l_tj[:, j]))
    r_l_corr = np.mean(r_l_corr_list)

    print('Residual Correlation: ', r_l_corr)

# Begin Simulation #################################################################################################
t_list = np.arange(0, 30+dt_imu, dt_imu)
t_gps_list = np.arange(0, 30+dt_gps, dt_gps)

N_realization = 1000
P_avg_list, e_avg_list, ortho_e_and_e_est_list, residual_all_realization = simulate_N_realizations(t_list, N_realization)

# Test one realization and visualize KF performance
x_true_list, delta_x_est_list, P_list, error_l, r_l = simulate_one_realization(t_list)

# Plotting Results
plot_one_realization_result(t_list, error_l, P_list, P_avg_list)
plot_avg_estimation_error(t_list, e_avg_list)
plot_orthogonality_for_state_est_and_est_error(t_list, ortho_e_and_e_est_list)
print_residual_correlation(t_gps_list, residual_all_realization)

plt.show()
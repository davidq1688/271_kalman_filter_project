import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Statistics and Parameters:
omega = 0.2  # angular frequency for the true acceleration profile

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

# Truth Model:
# def true_state_continuous(t, delta_x0):
#     t = np.array(t).reshape((-1,))
#     delta_x0 = delta_x0.reshape((-1,))
#     p0 = delta_x0[0] + p0_bar
#     v0 = delta_x0[1] + v0_bar
#     ba0 = delta_x0[2]

#     a = 10*np.sin(omega*t)
#     v = v0 + a/omega - a/omega*np.cos(omega*t)
#     p = p0 + (v0 + a/omega)*t - a/(omega**2)*np.sin(omega*t)
#     ba_list = ba0 * np.ones(np.shape(p))
#     return np.array([[p], [v], [ba_list]])

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

    # ba0 = ba_bar + np.sqrt(ba_var)*np.random.randn()
    # delta_v0 = np.sqrt(v0_var)*np.random.randn()
    # delta_p0 = np.sqrt(p0_var)*np.random.randn()
    delta_x0 = np.array([0, 0, 0]).reshape((-1,1))

    P0 = np.array([[p0_var, 0,0],
                   [0, v0_var, 0],
                   [0, 0, ba_var]])
    return delta_x0_true, delta_x0, P0


# Simulate one realization, initialize with random true state and random estimated state
# Output: delta_x_true_list (Nx3 ndarray)
# Output: delta_x_est_list (Nx3 ndarray)
# Output: P_list (list of N 3x3 P matrix)
# Output: state_error_list (Nx3 ndarray)
def simulate_one_realization(time_list, gps_interval=40):
    # initialize system true state, estimate state, and state covariance
    delta_x0_true, delta_x0_est, P0 = initialize()
    
    # state [dp_e = pe - pc, dv_e = ve - vc, ba]
    delta_x_true_list = []
    delta_x_est_list = []
    P_list = []
    x_true_list = []

    delta_x_true_list.append(delta_x0_true)
    delta_x_est_list.append(delta_x0_est)
    P_list.append(P0)
    x_true_list.append(delta_x0_true + np.array([[p0_bar],[v0_bar],[0]]))

    for k in range(len(time_list)-1):
        # imu noise
        w = generate_imu_noise_w()
        # true state propagate
        delta_x_true = PHI@delta_x_true_list[k] + GAMMA*w
        delta_x_true_list.append(delta_x_true)

        # Apriori prediction
        delta_x_pred = PHI@delta_x_est_list[k]
        M = PHI@P_list[k]@PHI.T + GAMMA@GAMMA.T*W

        # Aposteriori Estimation
        if k % gps_interval == 0:
            # measurement obtained
            eta = generate_gps_noise_eta()
            delta_z = gps_meas_update(delta_x_true, eta)
            # KF meas update: x_est = x_pred + K*(z - H*x_pred), K = P*H.T*inv(V)
            P = M - M@H.T@np.linalg.inv(H@M@H.T + V_gps)@H@M
            delta_x_est = delta_x_pred + P@H.T@np.linalg.inv(V_gps)@(delta_z - H@delta_x_pred)
        else:
            # no measurement obtained
            P = M
            delta_x_est = delta_x_pred
        
        P_list.append(P)
        delta_x_est_list.append(delta_x_est)

    delta_x_true_list = np.array(delta_x_true_list).reshape((-1, 3))
    delta_x_est_list = np.array(delta_x_est_list).reshape((-1, 3))
    state_error_list = delta_x_true_list - delta_x_est_list
    return delta_x_true_list, delta_x_est_list, P_list, state_error_list

# visualization functions
def plot_one_realization_result(t_list, delta_x_true_list, delta_x_est_list, P_list, P_avg_list):
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
    plt.plot(t_list, delta_x_true_list[:, 0]-delta_x_est_list[:, 0], label='pos estimation error')
    plt.plot(t_list, sigma_p_list, 'r', label='1-sigma bound')
    plt.plot(t_list, -1*np.array(sigma_p_list), 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.ylim([-1, 1])
    plt.legend()
    plt.title('Position Estimation Error')
    plt.grid()

    plt.subplot(132)
    plt.plot(t_list, delta_x_true_list[:, 1]-delta_x_est_list[:, 1], label='vel estimation error')
    plt.plot(t_list, sigma_v_list, 'r', label='1-sigma bound')
    plt.plot(t_list, -1*np.array(sigma_v_list), 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.ylim([-0.1, 0.1])
    plt.legend()
    plt.title('Velocity Estimation Error')
    plt.grid()

    plt.subplot(133)
    plt.plot(t_list, delta_x_true_list[:, 2]-delta_x_est_list[:, 2], label='bias estimation error')
    plt.plot(t_list, sigma_ba_list, 'r', label='1-sigma bound')
    plt.plot(t_list, -1*np.array(sigma_ba_list), 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('bias (m/s/s)')
    plt.ylim([-0.05, 0.05])
    plt.legend()
    plt.title('IMU bias Estimation Error')
    plt.grid()
    # plt.show()
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
    plt.title('P_11')
    plt.grid()
    plt.subplot(332)
    plt.plot(t_list, P12_error)
    plt.title('P_12')
    plt.grid()
    plt.subplot(333)
    plt.plot(t_list, P13_error)
    plt.title('P_13')
    plt.grid()
    plt.subplot(334)
    plt.plot(t_list, P21_error)
    plt.title('P_21')
    plt.grid()
    plt.subplot(335)
    plt.plot(t_list, P22_error)
    plt.title('P_22')
    plt.grid()
    plt.subplot(336)
    plt.plot(t_list, P23_error)
    plt.title('P_23')
    plt.grid()
    plt.subplot(337)
    plt.plot(t_list, P31_error)
    plt.title('P_31')
    plt.grid()
    plt.subplot(338)
    plt.plot(t_list, P32_error)
    plt.title('P_32')
    plt.grid()
    plt.subplot(339)
    plt.plot(t_list, P33_error)
    plt.title('P_33')
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
    # plt.plot(t_list, sigma_p_list, 'r', label='1-sigma bound')
    # plt.plot(t_list, -1*np.array(sigma_p_list), 'r')
    plt.xlabel('Time (s)')
    plt.title('Average Position Error')
    plt.grid()

    plt.figure()
    plt.plot(t_list, error_avg_list[1, :], label='avg vel est error')
    # plt.plot(t_list, sigma_v_list, 'r', label='1-sigma bound')
    # plt.plot(t_list, -1*np.array(sigma_v_list), 'r')
    plt.xlabel('Time (s)')
    plt.title('Average Velocity Error')
    plt.grid()

    plt.figure()
    plt.plot(t_list, error_avg_list[2, :], label='avg bias est error')
    # plt.plot(t_list, sigma_ba_list, 'r', label='1-sigma bound')
    # plt.plot(t_list, -1*np.array(sigma_ba_list), 'r')
    plt.xlabel('Time (s)')
    plt.title('Average Bias Error')
    plt.grid()

# Begin Simulation #################################################################################################
t_list = np.arange(0, 30+dt_imu, dt_imu)
N_realization = 1000

# Test one realization and visualize KF performance
delta_x_true_list, delta_x_est_list, P_list, error_l = simulate_one_realization(t_list)
# plot_one_realization_result(t_list, delta_x_true_list, delta_x_est_list, P_list)

# Ensamble of Realizations
P_list_all_realization = []  # N_realization x len(t_list) x np.array(3 x 3)
e_l_all_realization = []  # shape: N_realization x np.array(len(t_list) x 3)
error_p_all_realization = []  # shape: N_realization x len(t_list)
error_v_all_realization = []
error_b_all_realization = []

for i in tqdm(range(N_realization), desc='Simulating Realizations'):
    # simulate one realization
    delta_x_true_list, delta_x_est_list, P_list, error_l = simulate_one_realization(t_list)

    # save results
    error_p = np.array(error_l[:, 0]).reshape((-1,))
    error_v = np.array(error_l[:, 1]).reshape((-1,))
    error_b = np.array(error_l[:, 2]).reshape((-1,))

    error_p_all_realization.append(error_p)
    error_v_all_realization.append(error_v)
    error_b_all_realization.append(error_b)

    P_list_all_realization.append(P_list)
    e_l_all_realization.append(error_l)

error_p_avg = np.mean(error_p_all_realization, axis=0)  # avg of pos_error[time]
error_v_avg = np.mean(error_v_all_realization, axis=0)  # avg of vel_error[time]
error_b_avg = np.mean(error_b_all_realization, axis=0)  # avg of bias_error[time]
e_avg_list = np.array([error_p_avg, error_v_avg, error_b_avg])  # shape: 3xlen(t_list)

# Compute P_ave
P_avg_list = []
for i in tqdm(range(len(t_list)), desc='Computing P_avg[time]'):
    error_l_t = [e_l[i, :].reshape((-1,)) for e_l in e_l_all_realization]
    error_l_t = np.array(error_l_t).T  # shape: 3xN_realization
    temp = error_l_t - e_avg_list[:, i].reshape((-1, 1))  # shape: 3xN_realization
    P_t_list = []
    for j in range(N_realization):
        P_t_list.append(np.outer(temp[:, j], temp[:, j]))
    P_avg_list.append(np.mean(P_t_list, axis=0))

# Plotting Results
plot_one_realization_result(t_list, delta_x_true_list, delta_x_est_list, P_list, P_avg_list)
plot_avg_estimation_error(t_list, e_avg_list)

plt.show()
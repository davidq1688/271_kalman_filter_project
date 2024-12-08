import numpy as np
import matplotlib.pyplot as plt

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

    ba0 = ba_bar + np.sqrt(ba_var)*np.random.randn()
    delta_v0 = np.sqrt(v0_var)*np.random.randn()
    delta_p0 = np.sqrt(p0_var)*np.random.randn()
    delta_x0 = np.array([delta_p0, delta_v0, ba0]).reshape((-1,1))

    P0 = np.array([[p0_var, 0,0],
                   [0, v0_var, 0],
                   [0, 0, ba_var]])
    return delta_x0_true, delta_x0, P0

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

    return delta_x_true_list, delta_x_est_list, P_list

t_list = np.arange(0, 30+dt_imu, dt_imu)
N_realization = 1000

delta_x_true_list, delta_x_est_list, P_list = simulate_one_realization(t_list)

# visualization
delta_x_true_list = np.array(delta_x_true_list).reshape((-1, 3))
delta_x_est_list = np.array(delta_x_est_list).reshape((-1, 3))
sigma_p_list = []
sigma_v_list = []
sigma_ba_list = []
for i in range(len(t_list)):
    P_i = P_list[i]
    sigma_p_list.append(np.sqrt(P_i[0,0]))
    sigma_v_list.append(np.sqrt(P_i[1,1]))
    sigma_ba_list.append(np.sqrt(P_i[2,2]))

plt.figure()
# plt.plot(t_list, delta_x_true_list[:, 0], label='True Position', linestyle='dashed')
# plt.plot([m for m in measurements if m is not None], 'o', label='Measurements (Sparse)')
# plt.plot(t_list, delta_x_est_list[:, 0], label='Estimated Position', linewidth=2)
plt.plot(t_list, delta_x_true_list[:, 0]-delta_x_est_list[:, 0], label='pos estimation error')
plt.plot(t_list, sigma_p_list, 'r', label='1-sigma bound')
plt.plot(t_list, -1*np.array(sigma_p_list), 'r')
plt.xlabel('Time Step (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.title('Position Estimation Error')
plt.grid()

plt.figure()
plt.plot(t_list, delta_x_true_list[:, 1]-delta_x_est_list[:, 1], label='vel estimation error')
plt.plot(t_list, sigma_v_list, 'r', label='1-sigma bound')
plt.plot(t_list, -1*np.array(sigma_v_list), 'r')
plt.xlabel('Time Step (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.title('Velocity Estimation Error')
plt.grid()

plt.figure()
plt.plot(t_list, delta_x_true_list[:, 2]-delta_x_est_list[:, 2], label='bias estimation error')
plt.plot(t_list, sigma_ba_list, 'r', label='1-sigma bound')
plt.plot(t_list, -1*np.array(sigma_ba_list), 'r')
plt.xlabel('Time Step (s)')
plt.ylabel('bias (m/s/s)')
plt.legend()
plt.title('IMU bias Estimation Error')
plt.grid()
plt.show()
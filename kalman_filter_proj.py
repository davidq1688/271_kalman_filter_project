import numpy as np
import matplotlib.pyplot as plt

# Statistics and Parameters:
omega = 0.2  # angular frequency for the true acceleration profile

# IMU
imu_fs = 200  # IMU sample rate (Hz)
dt_imu = 1/imu_fs
W = 0.0004  # IMU meas. noise variance (m/s/s)**2

# bias apriori statistics
ba_bar = 0
ba_var = 0.01  # (m/s/s)**2
# ba = ba_bar + np.sqrt(ba_var)*np.random.randn()

# initial vel apriori statistics
v0_bar = 100  # (m/s)
v0_var = 1  # (m/s)**2
# v0 = v0_bar + np.sqrt(v0_var)*np.random.randn()

# initial pos apriori statistics
p0_bar = 0
p0_var = 10
# p0 = p0_bar + np.sqrt(p0_var)*np.random.randn()

# GPS
V_eta_p = 1  # variance of x meas. noise (m)**2
V_eta_v = (4/100)**2  # variance of v meas. noise (m/s)**2
V_gps = np.array([[V_eta_p, 0],
                  [0, V_eta_v]])
gps_fs = 5
gps_meas_interval = int(imu_fs/gps_fs)

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

# Truth Model:
def accel(t):
    return np.array(10*np.sin(omega*t)).reshape((-1,))

def vel(t, v0):
    a = accel(t)
    return v0 + a/omega - a/omega*np.cos(omega*t)

def pos(t, v0, p0):
    a = accel(t)
    return p0 + (v0 + a/omega)*t - a/(omega**2)*np.sin(omega*t)

def true_state_continuous(t, delta_x0):
    t = np.array(t).reshape((-1,))
    delta_x0 = delta_x0.reshape((-1,))
    p0 = delta_x0[0] + p0_bar
    v0 = delta_x0[1] + v0_bar
    ba0 = delta_x0[2]
    v = vel(t, v0)
    p = pos(t, v0, p0)
    ba_list = ba0 * np.ones(np.shape(p))
    return np.array([[p], [v], [ba_list]]).reshape((-1,1))

# IMU Model: (v_c0 = v0_bar, x_c0 = x0_bar)
v_c0 = v0_bar
p_c0 = p0_bar
def generate_imu_noise_w():
    w = np.sqrt(W)*np.random.randn()  # imu process noise
    return w

def a_c(current_a, ba, w):
    return current_a + ba + w

def v_c_step(a_current, v_c_current):
    return v_c_current + a_current*dt_imu

def p_c_step(a_current, v_c_current, p_c_current):
    return p_c_current + v_c_current*dt_imu + a_current*dt_imu**2/2

# Dynamic Model: (x_e0 = x0, v_e0 = v0)
def v_e(ve_current, a_current):
    return ve_current + a_current*dt_imu

def p_e(pe_current, ve_current, a_current):
    return pe_current + ve_current*dt_imu + a_current*dt_imu**2/2

# Discrete Linear System: state x = [delta_p_e, delta_v_e, ba], noise w = [w]
# x_k+1 = PHI*x_k + GAMMA*w
# delta_p_e = p_e - p_c, delta_v_e = v_e - v_c
PHI = np.array([[1, dt_imu, -dt_imu**2/2],
                [0, 1, -dt_imu],
                [0, 0, 1]])  # System State Matrix
GAMMA = np.array([[-dt_imu**2/2],
                  [-dt_imu],
                  [0]])  # System noise Matrix

# GPS Measurement Update
def generate_gps_noise_eta():
    eta_p = V_eta_p*np.random.randn()
    eta_v = V_eta_v*np.random.randn()
    return np.array([eta_p, eta_v]).reshape((-1,1))

def gps_meas_update(current_state_true, eta):
    current_state_true = current_state_true.reshape((-1,1))
    eta = eta.reshape((-1,1))
    z = current_state_true[0:2] + eta
    return z  # col vector [z_p, z_v]

# [delta_z_p, delta_z_v].T = [delta_p, delta_v].T + [eta_p, eta_v].T
H = np.array([[1, 0, 0],
              [0, 1, 0]])

def simulate_one_realization(time_list, gps_interval=40):
    # initialize system true state, estimate state, and state covariance
    delta_x0_true, delta_x0_est, P0 = initialize()
    
    # state [dp_e = pe - pc, dv_e = ve - vc, ba]
    delta_x_true_list = []
    delta_x_est_list = []
    P_list = []

    delta_x_true_list.append(delta_x0_true)
    delta_x_est_list.append(delta_x0_est)
    P_list.append(P0)

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
delta_x_true_list = np.array(delta_x_true_list).reshape((3, -1))
delta_x_est_list = np.array(delta_x_est_list).reshape((3, -1))

plt.figure(figsize=(12, 6))
plt.plot(t_list, delta_x_true_list[0, :], label='True Position', linestyle='dashed')
# plt.plot([m for m in measurements if m is not None], 'o', label='Measurements (Sparse)')
plt.plot(delta_x_est_list[0, :], label='Estimated Position', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.title('Kalman Filter with Sparse Measurements')
plt.grid()
plt.show()
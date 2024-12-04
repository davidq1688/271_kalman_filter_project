import numpy as np

# Statistics and Parameters:
omega = 0.2  # angular frequency for the true acceleration profile

# IMU
imu_fs = 200  # IMU sample rate (Hz)
dt_imu = 1/imu_fs
V = 0.0004  # IMU meas. noise variance (m/s/s)**2

# bias apriori statistics
ba_bar = 0
ba_var = 0.01  # (m/s/s)**2
ba = ba_bar + np.sqrt(ba_var)*np.random.randn()

# initial vel apriori statistics
v0_bar = 100  # (m/s)
v0_var = 1  # (m/s)**2
v0 = v0_bar + np.sqrt(v0_var)*np.random.randn()

# initial pos apriori statistics
x0_bar = 0
x0_var = 10
x0 = x0_bar + np.sqrt(x0_var)*np.random.randn()

# GPS
eta_x = 1  # variance of x meas. noise (m)**2
eta_v = (4/100)**2  # variance of v meas. noise (m/s)**2

# Truth Model:
def accel(t):
    return 10*np.sin(omega*t)

def vel(t):
    a = accel(t)
    return v0 + a/omega - a/omega*np.cos(omega*t)

def pos(t):
    a = accel(t)
    return x0 + (v0 + a/omega)*t - a/(omega**2)*np.sin(omega*t)

# IMU Model:
def a_c(t):
    w = np.sqrt(V)*np.random.randn()  # imu meas. noise
    return accel(t) + ba + w

def v_c_step(a_current, v_c_current):
    return v_c_current + a_current*dt_imu

def x_c_step(a_current, v_c_current, x_c_current):
    return x_c_current + v_c_current*dt_imu + a_current*dt_imu**2/2

# Dynamic Model: (x_e0 = x0, v_e0 = v0)
def v_e(ve_current, a_current):
    return ve_current + a_current*dt_imu

def x_e(xe_current, ve_current, a_current):
    return xe_current + ve_current*dt_imu + a_current*dt_imu**2/2

# Discrete Linear System: state [delta_x_e, delta_v_e, ba], noise [w]
# delta_x_e = x_e - x_c, delta_v_e = v_e - v_c
PHI = np.array([[1, dt_imu, -dt_imu**2/2],
                [0, 1, -dt_imu],
                [0, 0, 1]])  # System State Matrix
GAMMA = np.array([[-dt_imu**2/2],
                  [-dt_imu],
                  [0]])  # System noise Matrix
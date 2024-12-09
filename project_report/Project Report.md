## Abstract

This report presents the implementation and analysis of a Kalman Filter designed for an integrated Inertial Measurement Unit (IMU) and Global Positioning System (GPS) system. The system operates on a single-axis motion, propagates states based on sensor measurements, and incorporates noisy data to estimate position, velocity, and sensor bias. Simulation results demonstrate the filter's ability to converge to the true state and the effectiveness of uncertainty bounds derived from the covariance matrix.

## Theory

The Kalman Filter (KF) is a recursive state estimation algorithm for linear systems. It comprises two main stages: prediction and update. In the prediction step, the system state is propagated forward in time using a dynamic model. In the update step, new measurements are incorporated to correct the predicted state and reduce estimation uncertainty.

For this project, the system is modeled as follows:

- **State Vector:** $\delta x = [\delta p_e, \delta v_e, b_a]^T$, where $\delta p_e$ is position error, $\delta v_e$ is velocity error, and $b_a$ is IMU bias.
- **State Propagation:**

  \[
$x_{k+1} = \Phi x_k + \Gamma w_k$
  \]

  where \(\Phi\) and \(\Gamma\) are the state transition and noise matrices, and \(w_k\) is the process noise.

- **Measurement Model:**

  \[
z_k = Hx_k + \eta_k
  \]

  where \(H\) is the observation matrix, \(\eta_k\) is the measurement noise, and \(z_k\) represents GPS measurements of position and velocity.

The Kalman Filter equations for this setup are:

1. **Prediction:**
   \[
   \hat{x}_{k|k-1} = \Phi \hat{x}_{k-1|k-1}
   \]
   \[
   P_{k|k-1} = \Phi P_{k-1|k-1}\Phi^T + Q
   \]

2. **Update:**
   \[
   K_k = P_{k|k-1} H^T (HP_{k|k-1} H^T + R)^{-1}
   \]
   \[
   \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H\hat{x}_{k|k-1})
   \]
   \[
   P_{k|k} = (I - K_k H)P_{k|k-1}
   \]

## Results

### Simulation Setup

The IMU generates data at 200 Hz, while the GPS provides position and velocity updates at 5 Hz. Process noise and measurement noise are modeled with Gaussian distributions:

- IMU Noise Variance: \(0.0004\, \text{(m/s}^2\text{)}^2\)
- GPS Position Noise Variance: \(1\, \text{m}^2\)
- GPS Velocity Noise Variance: \((0.04\, \text{m/s})^2\)

The Kalman Filter was tested over a 30-second simulation with 100 realizations to analyze average error and covariance consistency.

### Key Findings

1. **Position and Velocity Estimation Errors:**
   The errors for position and velocity consistently converged to zero, as shown in the figures. The 1-sigma bounds derived from the covariance matrix accurately captured the estimation uncertainty.

2. **Bias Estimation:**
   The filter successfully tracked the IMU bias over time, demonstrating its adaptability.

3. **Residual Analysis:**
   Residuals were uncorrelated, validating the filter’s assumptions.

4. **Orthogonality:**
   The orthogonality between estimation errors and the state estimates was confirmed, highlighting the filter’s optimality.

## Analysis

The Kalman Filter’s performance was assessed using several metrics:

- **Error Dynamics:** The error in position and velocity decreased exponentially, aligning with theoretical expectations.
- **Covariance Consistency:** The estimated covariance bounds encompassed the true errors, indicating appropriate tuning of process and measurement noise parameters.
- **Residual Statistics:** Residuals followed Gaussian distributions with zero mean, reinforcing the validity of the noise assumptions.

Challenges observed included sensitivity to initial conditions and noise statistics. Robustness to parameter misestimation remains an area for further investigation.

## Conclusion

The implemented Kalman Filter effectively estimated position, velocity, and bias for a single-axis motion system. Simulation results verified the filter’s accuracy and theoretical consistency. Future work may extend this implementation to multi-dimensional motion and incorporate adaptive noise modeling for enhanced robustness.


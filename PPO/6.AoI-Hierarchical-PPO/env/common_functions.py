import numpy as np
import os
import math

def calculate_uav_energy_consumption(dx, dy, dz, delta_t):
    """
    Args:
        :param dx: x轴移动距离
        :param dy: y轴移动距离
        :param dz: z轴移动距离
        :param delta_t: The time duration of a time slot
    :return:
        Energy consumption of UAV in current time slot
    :Reference:
        Energy Minimization for Wireless Communication With Rotary-Wing UAV
    """
    # 将dx，dy和dz转化成水平和垂直的飞行速度
    v_horizontal = np.sqrt(dx**2 + dy**2)/delta_t
    v_vertical = abs(dz)/delta_t

    W = 20                             # aircraft weight in Newton, kg
    rho = 1.225                        # air density, kg/m^3
    R = 0.4                            # blade radius in meter, m
    A = np.pi * R ** 2                 # rotor disc area
    omega = 300                        # blade angular velocity in rad/s
    V_tip = omega * R                  # tip speed of the rotor blade
    b = 4                              # number of blades
    c = 0.0157                         # blade or aerofoil chord length
    sigma = b * c / (np.pi * R)        # rotor solidity
    S_FP = 0.0151                      # Fuselage equivalent flat plate area in m^2
    d_0 = S_FP / sigma / A             # fuselage drag ratio
    k = 0.1                            # incremental correction factor to induced power
    v_0 = np.sqrt(W / (2 * rho * A))   # mean rotor induced velocity in hover
    delta = 0.012                      # profile drag coefficient
    g = 9.8                            # gravity acceleration, m/s^2

    P_blade = (delta / 8) * rho * sigma * A * (omega ** 3) * (R ** 3)      # blade profile power in hovering
    P_induced = (1 + k) * (W ** 1.5) / np.sqrt(2 * rho * A)                # induced power in hovering

    # propulsion energy consumption
    P_t = (P_blade * (1 + 3 * (v_horizontal ** 2) / (V_tip ** 2)) +
           P_induced * np.sqrt(np.sqrt(1 + v_horizontal ** 4/(4 * v_0 ** 4)) - v_horizontal ** 2/(2 * v_0 ** 2)) +
           1/2 * d_0 * rho * sigma * A * v_horizontal ** 3 + W * g * v_vertical
           ) * delta_t
    return P_t

def channel_power_gain(loca_transmitter, loca_receiver):
    """
    Args:
        :param loca_transmitter: The location of transmitter
        :param loca_receiver: The location of receiver
    return:
        The channel power gain between the transmitter and the receiver
    Reference:
        Mobile Unmanned Aerial Vehicles (UAVs) for Energy-Efficient Internet of Things Communications
    """
    a = 11.95
    b = 0.14
    f = 2e9
    wave_length = 3e8 / f
    k_o = (4 * np.pi/wave_length) ** 2
    eta_los = 2.00             # attenuation factor of los
    eta_non_los = 199.53       # attenuation factor of non los

    dist_vert = abs(loca_transmitter[2] - loca_receiver[2])
    dist_horizon = np.sqrt((loca_transmitter[0] - loca_receiver[0])**2 + (loca_transmitter[1] - loca_receiver[1])**2)
    dist_total = np.sqrt((loca_transmitter[0] - loca_receiver[0])**2 + (loca_transmitter[1] - loca_receiver[1])**2 +
                         (loca_transmitter[2] - loca_receiver[2])**2)

    elevation = np.arctan(dist_vert / dist_horizon)  # arc in radians
    angle = np.degrees(elevation)                  # Convert to degrees

    temp = np.exp((-b) * (angle - a))
    p_los = 1/(1 + a * temp)
    p_non_los = 1 - p_los

    path_loss = (p_los * eta_los + p_non_los * eta_non_los) * k_o * dist_total ** 2
    channel_gain = 1 / path_loss
    return channel_gain


def calculate_rate_device_UAV(User, UAV, band_fraction, power):
    """
    Args:
        :param User: The entity of ground device
        :param UAV: The entity of UAV
        :param band_fraction: The bandwidth fraction allocated by UAV to the current user
    return:
        The transmission rate between the transmitter and the receiver
    Reference:
        Energy Efficiency Maximization for UAV-Enabled Hybrid Backscatter-Harvest-ThenTransmit Communications
    """
    band_width = 20e6          # 20MHz
    spectral_density = 10 ** (-13)         # -130dBm/Hz = 10^(-130/10) mW/Hz = 10^-13 W/Hz
    channel_gain = channel_power_gain(User.position, UAV.position)
    epsilon = 1e-9  # 一个很小的正数
    denominator = (band_width * band_fraction * spectral_density) + epsilon
    rate = band_fraction * band_width * np.log2(1 + (power * channel_gain) / denominator)
    return rate

"""
AfshariStewart_2016_Ds.py
Jason Motha
29/3/16

Edited by J. Hutchinson
20/4/2021

Provides the ground motion prediction equation for Significant duration
defined at the time from 5-95% of the Arias intensity

Reference: Kempton JJ, Stewart JP.  Prediction equations for significant duration
of earthquake ground motions considering site and near-source effects.  
Earthquake Spectra 2006, 22(4), pp985-1013.

This implementation only considers the acceleration-based measures, Ds5-75
and Ds5-95, i.e. the times between the 5% and 75%/95% of the arias
intensity buildup.

The constant values in the lists below are arranged as:
    1st value Ds575
    2nd value Ds595
    3rd value Ds2080

Input Variables:
 
 rrup             = Source-to-site distance (km) (closest distance)
 siteprop      = properties of site (soil etc)
                   Rrup    - Source-to-site distance
                   Vs30    - shear wave velocity of upper 30m
                   z1.5    - depth to 1500m/s shear wave velocity (in m)
                   defn =0 sign duration 5-75% arias intenstiy integral
                       =1 sign duration 5-95% arias intensity integral (default)
                       =2 sign duration 20-80% arias intensity integral
 faultprop     = properties of fault
                    Mw - Moment magnitude (Mw)


Output Variables:
 Ds           = median Ds  
 sigma_Ds     = lognormal standard deviation in Ds
                 %sigma_Ds(1) = total std
                 %sigma_Ds(2) = interevent std
                 %sigma_Ds(3) = intraevent std

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""
import numpy as np


beta = 3.2
Mstar = 6.0

M1 = [5.35, 5.2, 5.2]
M2 = [7.15, 7.4, 7.4]

b2 = [0.9011, 0.9443, 0.7414]
b3 = [-1.684, -3.911, -3.164]

c1 = [0.1159, 0.3165, 0.0646]
c2 = [0.1065, 0.2539, 0.0865]
c3 = [0.0682, 0.0932, 0.0373]
c5 = [0.4, 0.4, 0.4]

c4 = [-0.2246, -0.3183, -0.4237]

R1 = 10
R2 = 50
R3 = 300

V1 = 600
Vref = [368.2, 369.9, 369.6]
sigma_z1ref = 200

tau1 = [0.28, 0.25, 0.30]
tau2 = [0.25, 0.19, 0.19]
phi1 = [0.54, 0.43, 0.56]
phi2 = [0.41, 0.35, 0.45]


def estimate_z1p0(vs30):
    return (
        np.exp(28.5 - 3.82 / 8.0 * np.log(vs30**8 + 378.7**8)) / 1000.0
    )  # CY08 estimate in KM


def Afshari_Stewart_2016_Ds(Mw, rrup, vs30, faultstyle, im):
    z1p0 = estimate_z1p0(vs30)

    if im == "Ds575":
        i = 0
    elif im == "Ds595":
        i = 1
    elif im == "Ds2080":
        i = 2
    else:
        print("Invalid IM specified")
        exit()

    M0 = 10 ** (1.5 * Mw + 16.05)

    if faultstyle == "normal":
        b0 = [1.555, 2.541, 1.409]
        b1 = [4.992, 3.170, 4.778]
    elif faultstyle == "reverse":
        b0 = [0.7806, 1.612, 0.7729]
        b1 = [7.061, 4.536, 6.579]
    elif faultstyle == "strikeslip":
        b0 = [1.279, 2.302, 0.8804]
        b1 = [5.578, 3.467, 6.188]
    else:
        b0 = [1.28, 2.182, 0.8822]
        b1 = [5.576, 3.628, 6.182]

    if Mw <= M2[i]:
        delta_sigma = np.exp(b1[i] + b2[i] * (Mw - Mstar))
    else:
        delta_sigma = np.exp(b1[i] + b2[i] * (M2[i] - Mstar) + b3[i] * (Mw - M2[i]))

    f0 = 4.9 * 10**6 * beta * (delta_sigma / M0) ** (1.0 / 3.0)

    if Mw > M1[i]:
        Fe = 1 / f0
    else:
        Fe = b0[i]

    if rrup <= R1:
        Fp = c1[i] * rrup
    elif rrup <= R2:
        Fp = c1[i] * R1 + c2[i] * (rrup - R1)
    else:
        Fp = c1[i] * R1 + c2[i] * (R2 - R1) + c3[i] * (rrup - R2)

    # Magnitude-distance function added by J. Hutchinson, may need more tweaking
    if Mw >= 5:
        Fp = Fp + (0.124 * rrup + 20.46) * (Mw - 5)

    # California
    MuZ1 = np.exp(
        -7.15 / 4 * np.log((vs30**4 + 570.94**4) / (1360**4 + 570.94**4))
        - np.log(1000)
    )

    delta_z1 = z1p0 - MuZ1

    if delta_z1 <= sigma_z1ref:
        FsigmaZ1 = c5[i] * delta_z1
    else:
        FsigmaZ1 = c5[i] * sigma_z1ref

    if vs30 <= V1:
        Fs = c4[i] * np.log(vs30 / Vref[i]) + FsigmaZ1
    else:
        Fs = c4[i] * np.log(V1 / Vref[i]) + FsigmaZ1

    Ds = np.exp(np.log(Fe + Fp) + Fs)

    if Mw < 6.5:
        tau_M = tau1[i]
    elif Mw < 7:
        tau_M = tau1[i] + (tau2[i] - tau1[i]) * ((Mw - 6.5) / (7 - 6.5))
    else:
        tau_M = tau2[i]

    if Mw < 5.5:
        phi_M = phi1[i]
    elif Mw < 5.75:
        phi_M = phi1[i] + (phi2[i] - phi1[i]) * ((Mw - 5.5) / (5.75 - 5.5))
    else:
        phi_M = phi2[i]

    total_sigma_Ds = np.sqrt(tau_M**2 + phi_M**2)
    sigma_Ds = [total_sigma_Ds, tau_M, phi_M]

    return Ds, sigma_Ds

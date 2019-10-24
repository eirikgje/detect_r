import camb
import healpy
import numpy as np
from scipy.optimize import fmin_powell

h = 6.62607004e-34
k_b = 1.38064852e-23
c = 2.99792458e8
t_cmb = 2.7255


def compute_map(r, lmax=1500, nside=512, seed=None):
    np.random.seed(seed)
    cp = camb.set_params(r=r, WantCls=True, lmax=lmax, H0=67,
                         Want_CMB_lensing=False, DoLensing=False,
                         WantScalars=False)
    res = camb.get_results(cp)
    cls = res.get_cmb_power_spectra(CMB_unit='muK')
    cls = np.transpose(cls['total'])
    ells = np.arange(len(cls[0, :]))
#    return cls
    cls[:, 1:] = cls[:, 1:] / (ells[1:] * (ells[1:] + 1))
    outmap = healpy.synfast(cls, nside, new=True, pol=True)
    outmap *= 2.7255 * 1e-6
    return outmap


def get_diff_map(r, dr, seed=None):
    map1 = compute_map(r, seed=seed)
    map2 = compute_map(r + dr, seed=seed)
    return map1 - map2, map1, map2


def find_max_amp(target_diff_kcmb, beta1, beta2, t_d1, t_d2, target_freq=100e9,
                 start_freq=400e9, reference_freq=100e9):
    target_diff_rj = target_diff_kcmb * k_cmb2k_rj(np.array([target_freq]), np.array([1.0]), target_freq)
    amplitude_rj = target_diff_rj / np.abs(
        (target_freq / start_freq) ** (beta1 - 2) * (blackbody(t_d1, target_freq) / blackbody(t_d1, start_freq)) -
        (target_freq / start_freq) ** (beta2 - 2) * (blackbody(t_d2, target_freq) / blackbody(t_d2, start_freq)))
    amplitude = amplitude_rj / k_cmb2k_rj(np.array([start_freq]), np.array([1.0]), start_freq)
    optical_depth = rjamp_to_optical_depth(amplitude_rj, beta1, t_d1, start_freq, reference_freq)
    amplitude_rj_ref = optical_depth / rjamp_to_optical_depth(1.0, beta1, t_d1, reference_freq, reference_freq)
    return optical_depth, amplitude_rj, amplitude, amplitude_rj_ref


def rjamp_to_optical_depth(amplitude_krj, beta, t_d, freq, reference_freq):
    return (amplitude_krj / ((freq / reference_freq) ** beta * 
                             blackbody(t_d, freq)) / 
            (c ** 2 / (2 * k_b * freq ** 2)))


def blackbody(temperature, freq):
    return 2 * h * freq ** 3 / c ** 2 / (np.exp(h * freq / (k_b * temperature)) - 1)


#def modified_blackbody(opacity, beta, t_d, freq_0, freq):
#    return opacity * (freq/freq_0) ** beta * blackbody(t_d, freq)
#

def k_cmb2k_rj(freqs, bandpass, central_freq):
    mjysr2k_rj = c ** 2 / 2e20 / central_freq ** 2 / k_b
    a = np.exp(h * freqs / k_b / t_cmb)
    b = 1 / (a - 1)
    d = (2 * h ** 2 * freqs ** 4) / (c ** 2 * k_b * t_cmb ** 2)
    y2 = a * d * b ** 2
    yNumCMB = y2
    y4 = central_freq / freqs
    yDenMJ = y4
    if len(bandpass) == 1:
        intNumCMB = yNumCMB
        intDenMJ = yDenMJ / 1e20
    else:
        intNumCMB = np.trapz(yNumCMB * bandpass, freqs)
        intDenMJ = np.trapz(yDenMJ * bandpass, freqs) / 1e20
    k_cmb2mjysr = intNumCMB / intDenMJ

    factor = k_cmb2mjysr * mjysr2k_rj
    return factor


# The part below is for simulating and fitting models with more than one cloud
def n_cloud_model(freqs, reference_freq, opacity_mean, opacity_sigma, angle_mean,
                  angle_sigma, beta_mean, beta_sigma,
                  temperature_mean, temperature_sigma, num_clouds=2):
    total_amplitude = np.zeros(len(freqs))
    means = np.array([opacity_mean, angle_mean, beta_mean, temperature_mean])
    sigmas = np.array([opacity_sigma, angle_sigma, beta_sigma,
                       temperature_sigma])
    param_set = [np.random.normal(means, sigmas) for i in range(num_clouds)]
    for params in param_set:
        total_amplitude += (params[0] * np.cos(params[1]) *
            (freqs/reference_freq) ** params[2] * blackbody(params[3], freqs))
    return total_amplitude * 1e16, param_set


def fit_mbb(params0, datapoints, freqs, reference_freq):

    def mbb_model(params, freqs=freqs, reference_freq=reference_freq):
        return params[0] * (freqs/reference_freq) ** params[1] * blackbody(params[2], freqs)

    def residuals(params, datapoints=datapoints):
        guess = mbb_model(params) * 1e16
        return np.sqrt(np.sum((datapoints - guess) ** 2))

    return fmin_powell(residuals, params0, disp=True, ftol=1e-20, maxiter=1000000, maxfun=1000000)


def mbb_model(params, freqs=freqs, reference_freq=reference_freq):
    return params[0] * (freqs/reference_freq) ** params[1] * blackbody(params[2], freqs)


def run_simulation(num_clouds=2):
#    freqs = np.array([30e9, 44e9, 70e9, 100e9, 217e9, 353e9, 400e9])
    freqs = np.linspace(30e9, 1500e9)
    reference_freq = 100e9
    opacity_mean = 0.3
    opacity_sigma = 0.1
    angle_mean = 1e-2
    angle_sigma = 1e-3
    beta_mean = 1.5
    beta_sigma = 0.2
    temperature_mean = 20
    temperature_sigma = 5

    res = n_cloud_model(freqs, reference_freq, opacity_mean, opacity_sigma,
                        angle_mean, angle_sigma, beta_mean, beta_sigma,
                        temperature_mean, temperature_sigma,
                        num_clouds=num_clouds)
    fit = fit_mbb(np.array([opacity_mean, beta_mean, temperature_mean]), res[0], freqs, reference_freq)
    return fit, res[1]


def run_simulations(num_simulations=1000, num_clouds=2):
    fits = []
    ress = []
    for i in range(num_simulations):
        out = run_simulation(num_clouds=num_clouds)
        fits.append(out[0])
        ress.append(out[1])
    return np.array(fits), np.array(ress)


def extrapolate_


def estimate_beta_err_n_clouds(opacity_mean, opacity_sigma, angle_mean,
                               angle_sigma, beta_mean, beta_sigma,
                               temperature_mean, temperature_sigma,
                               start_freq=400e9, target_freq=100e9,
                               reference_freq=100e9, num_clouds=2,
                               freqs=np.array([30e9, 44e9, 70e9, 100e9, 217e9,
                                               353e9, 400e9, 545e9, 857e9])):
    model = n_cloud_model(np.array([target_freq, start_freq]), reference_freq,
                          opacity_mean, opacity_sigma, angle_mean, angle_sigma,
                          beta_mean, beta_sigma, temperature_mean,
                          temperature_sigma, num_clouds=num_clouds)
    fit = fit_mbb(np.array([opacity_mean, beta_mean, temperature_mean]),
                  model[0], freqs, reference_freq)
    beta_fit = fit[2]
    fit_amplitude = 

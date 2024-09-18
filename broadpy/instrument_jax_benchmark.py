from broadpy import InstrumentalBroadening, InstrumentalBroadeningJax
from broadpy.utils import load_example_data

import numpy as np
import jax.numpy as jnp

import time



# Load example data
wave, flux = load_example_data(wave_range=(2320.0, 2430.0))

IB = InstrumentalBroadening(wave, flux)
IBJ = InstrumentalBroadeningJax(wave, flux)

# Instrumental resolution
R = 1e5
fwhm = 2.998e5/R * np.ones_like(wave)
fwhm_jax = 2.998e5/R * jnp.ones_like(wave)

# Convolve with Gaussian kernel


n = 20
times = []
for i in range(n):
    start = time.time()
    y_lsf = IB(fwhm=fwhm, kernel='gaussian_variable')
    end = time.time()
    times.append(end-start)
print(f'Numpy: {np.mean(times[1:]):.3f} +- {np.std(times[1:]):.3f}')
# y_lsf = IB(fwhm=fwhm, kernel='gaussian_variable')
# y_lsf_jax = IBJ(fwhm=fwhm_jax, kernel='gaussian_variable')

times_jax = []
for i in range(n):
    start = time.time()
    y_lsf_jax = IBJ(fwhm=fwhm_jax, kernel='gaussian_variable')
    end = time.time()
    times_jax.append(end-start)
print(f'Jax: {np.mean(times_jax[1:]):.3f} +- {np.std(times_jax[1:]):.3f}')


print(np.allclose(y_lsf, y_lsf_jax, atol=1e-5))
print(f' TODO: test it on a GPU e.g. merwede')
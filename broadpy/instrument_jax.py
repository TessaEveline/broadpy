import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial

c = 2.998e5  # km/s
sqrt8ln2 = jnp.sqrt(8 * jnp.log(2))

# Helper functions outside the class
@jax.jit
def gaussian_profile(x, x0, sigma):
    """Gaussian function."""
    return jnp.exp(-0.5 * ((x - x0) / sigma) ** 2)

@jax.jit
def lorentz_profile(x, x0, gamma):
    """Lorentzian function."""
    return gamma / jnp.pi / ((x - x0) ** 2 + gamma ** 2)

# @jax.jit
# @partial(jax.jit, static_argnums=(2,))  # Treat truncate as static
def compute_gaussian_kernel(fwhm, spacing, truncate=4.0):
    sd = (fwhm / c) / sqrt8ln2 / spacing
    lw = jax.lax.floor(truncate * sd + 0.5).astype(int)  # Use lax.floor and ensure the result is an integer
    
    # Create dynamic range using lax.iota instead of jnp.arange
    kernel_x = jax.lax.iota(jnp.float32, 2 * lw + 1) - lw
    kernel = gaussian_profile(kernel_x, 0, sd)
    kernel /= jnp.sum(kernel)  # Normalize the kernel
    return kernel

# @jax.jit
def compute_gaussian_variable_kernel(fwhm, spacing, truncate=4.0):
    """Gaussian convolution kernel with variable FWHM."""
    sd = (fwhm / c) / sqrt8ln2 / spacing
    lw = jnp.floor(truncate * sd.max() + 0.5)
    x = jnp.arange(-lw, lw + 1)
    
    # Use vmap to vectorize the kernel calculation for each point in fwhm
    kernels = jit(vmap(lambda s: jnp.exp(-0.5 * (x / s) ** 2)))(sd)
    kernels /= kernels.sum(axis=1)[:, None]  # Normalize each kernel
    return kernels, lw

# @jax.jit
def compute_lorentz_kernel(gamma, spacing, truncate=4.0):
    """Lorentzian convolution kernel helper."""
    gamma_pixels = gamma / c / spacing
    lw = jnp.floor(truncate * gamma_pixels + 0.5)
    kernel_x = jnp.arange(-lw, lw + 1)
    kernel = lorentz_profile(kernel_x, 0, gamma_pixels)
    kernel /= jnp.sum(kernel)
    return kernel

# @jax.jit
def compute_voigt_kernel(fwhm, gamma, spacing, truncate=4.0):
    """Voigt convolution kernel helper."""
    _kernel_g = compute_gaussian_kernel(fwhm, spacing, truncate)
    _kernel_l = compute_lorentz_kernel(gamma, spacing, c, truncate)
    
    kernel = jax.scipy.signal.convolve(_kernel_g, _kernel_l, mode='same')
    kernel /= jnp.sum(kernel)
    return kernel

class InstrumentalBroadening:
    
    available_kernels = ['gaussian', 'lorentzian', 'voigt', 'gaussian_variable', 'auto']
    
    def __init__(self, x, y):
        self.x = jnp.asarray(x)  # Convert to JAX arrays
        self.y = jnp.asarray(y)  # Convert to JAX arrays
        assert len(self.x) == len(self.y), 'x and y should have the same length'
        # Mean spacing between wavelength values
        self.spacing = jnp.mean(2 * jnp.diff(self.x) / (self.x[1:] + self.x[:-1]))
        print(f' spacing = {self.spacing}')

    def __call__(self, res=None, fwhm=None, gamma=None, truncate=4.0, kernel='auto'):
        """Instrumental broadening function with kernel selection."""
        kernel = self.__read_kernel(res=res, fwhm=fwhm, gamma=gamma) if kernel == 'auto' else kernel
        assert kernel in self.available_kernels, f'Please provide a valid kernel: {self.available_kernels}'
        
        if kernel in ['gaussian', 'voigt']:
            fwhm = fwhm if fwhm is not None else (self.c / res)

        if kernel == 'gaussian':
            _kernel = compute_gaussian_kernel(fwhm, self.spacing, truncate)
        
        if kernel == 'lorentzian':
            assert gamma is not None, 'Please provide a gamma value for the Lorentzian kernel'
            _kernel = compute_lorentz_kernel(gamma, self.spacing, self.c, truncate)
        
        if kernel == 'voigt':
            assert gamma is not None, 'Please provide a gamma value for the Lorentzian kernel'
            _kernel = compute_voigt_kernel(fwhm, gamma, self.spacing, truncate)
        
        if kernel == 'gaussian_variable':
            assert isinstance(fwhm, (list, jnp.ndarray)), 'Please provide a list of FWHM values'
            assert len(fwhm) == len(self.x), 'FWHM list should have the same length as the wavelength array'
            _kernels, lw = compute_gaussian_variable_kernel(fwhm, self.spacing, truncate)
            # make lw of integral type compatible with jnp.pad
            lw = int(lw)
            y_pad = jnp.pad(self.y, (lw, lw), mode='reflect')
            y_matrix = jnp.array([y_pad[i:i + len(self.y)] for i in range(2 * lw + 1)]).T
            y_lsf = jnp.einsum('ij, ij->i', _kernels, y_matrix)
            return y_lsf

        # Use JAX's convolve function for 1D convolution
        y_lsf = jax.scipy.signal.convolve(self.y, _kernel, mode='same')
        return y_lsf
    
    def __read_kernel(self, res=None, fwhm=None, gamma=None):
        """Read the kernel type from the input parameters."""
        if res is not None:
            return 'gaussian'
        if fwhm is not None and gamma is None:
            return 'gaussian'
        if fwhm is None and gamma is not None:
            return 'lorentzian'
        if fwhm is not None and gamma is not None:
            return 'voigt'
        if fwhm is not None and isinstance(fwhm, (list, jnp.ndarray)):
            return 'gaussian_variable'
        raise ValueError(f'Please provide a valid kernel: {InstrumentalBroadening.available_kernels}')


if __name__=='__main__':
    from broadpy.utils import load_example_data
    wave, flux = load_example_data()
    R = 1e5
    fwhm = 2.998e5/R * jnp.ones_like(wave)
    
    IB = InstrumentalBroadening(wave, flux)
    y_lsf = IB(fwhm=fwhm, kernel='gaussian_variable')
    
    
import numpy as np
from scipy.signal import convolve 
from scipy.ndimage import convolve1d # suitable for 'same' output shape as input

class InstrumentalBroadening:
    
    c = 2.998e5 # km/s
    sqrt8ln2 = np.sqrt(8 * np.log(2))
    
    available_kernels = ['gaussian', 'lorentzian', 'voigt', 'gaussian_variable']
    
    def __init__(self, x, y):
        
        self.x = x # units of wavelength
        self.y = y # units of flux (does not matter)
        assert len(self.x) == len(self.y), 'x and y should have the same length'

        self.spacing = np.mean(2*np.diff(self.x) / (self.x[1:] + self.x[:-1]))
    
    def __call__(self, out_res=None, fwhm=None, gamma=None, kernel='voigt', truncate=4.0):
        '''Instrumental broadening
        
        provide either instrumental resolution lambda/delta_lambda or FWHM in km/s'''
        
        assert kernel in self.available_kernels, 'Please provide a valid kernel: gaussian, lorentzian, voigt'
            
        if kernel in ['gaussian', 'voigt']:
            fwhm = fwhm if fwhm is not None else (self.c / out_res)

        if kernel == 'gaussian':
            _kernel = self.gaussian_kernel(fwhm, truncate)
        
        if kernel == 'lorentzian':
            assert gamma is not None, 'Please provide a gamma value for the Lorentzian kernel'
            _kernel = self.lorentz_kernel(gamma, truncate)
        
        if kernel == 'voigt':
            assert gamma is not None, 'Please provide a gamma value for the Lorentzian kernel'        
            _kernel = self.voigt_kernel(fwhm, gamma, truncate)
            
        if kernel == 'gaussian_variable':
            assert isinstance(fwhm, (list, np.ndarray)), 'Please provide a list of FWHM values'
            assert len(fwhm) == len(self.x), 'FWHM list should have the same length as the wavelength array'
            _kernels, lw = self.gaussian_variable_kernel(fwhm, truncate)
            y_pad = np.pad(self.y, (lw, lw), mode='reflect')
            y_matrix = np.array([y_pad[i:i + len(self.y)] for i in range(2 * lw + 1)]).T
            y_lsf = np.einsum('ij, ij->i', _kernels, y_matrix)
            return y_lsf
            
        y_lsf = convolve1d(self.y, _kernel, mode='nearest')
        return y_lsf
    
    @classmethod
    def gaussian_profile(self, x, x0, sigma):
        '''Gaussian function'''
        return np.exp(-0.5 * ((x - x0) / sigma)**2)# / (sigma * np.sqrt(2*np.pi))
    
    @classmethod
    def lorentz_profile(self, x, x0, gamma):
        '''Lorentzian function'''
        return gamma / np.pi / ((x - x0)**2 + gamma**2)

    
    def gaussian_kernel(self, 
                        fwhm,
                        truncate=4.0, 
                        ):
        ''' Gaussian kernel
        
        Parameters
        ----------
        fwhm : float
            Full width at half maximum of the Gaussian kernel in km/s
        truncate : float
            Truncate the kernel at this many standard deviations from the mean (default: 4.0)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''
        # Adapted from scipy.ndimage.gaussian_filter1d        
        sd = (fwhm/self.c) / self.sqrt8ln2 / self.spacing
        lw = int(truncate * sd + 0.5)
    
        kernel_x = np.arange(-lw, lw+1)
        kernel = self.gaussian_profile(kernel_x, 0, sd)
        kernel /= np.sum(kernel)  # normalize the kernel
        return kernel
    
    def gaussian_variable_kernel(self, fwhm, truncate=4.0):
        ''' Gaussian kernel with variable FWHM
        
        Parameters
        ----------
        fwhm : array
            Full width at half maximum of the Gaussian kernel in km/s
        truncate : float
            Truncate the kernel at this many standard deviations from the mean (default: 4.0)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''
        sd = (fwhm/self.c) / self.sqrt8ln2 / self.spacing
        lw = int(truncate * sd.max() + 0.5)
        x = np.arange(-lw, lw + 1)
        
        # Use broadcasting to create a 2D array of Gaussian kernels
        kernels = np.exp(-0.5 * (x[None, :] / sd[:, None]) ** 2)
        kernels /= kernels.sum(axis=1)[:, None]
        return kernels, lw
        
    
    def lorentz_kernel(self, gamma, truncate=4.0):
        ''' Lorentzian kernel
        Parameters
        ----------
        gamma : float
            Full width at half maximum of the Lorentzian kernel in km/s
        truncate : float
            Extent of the kernel as a multiple of gamma (default: 5)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        ''' 
        gamma_pixels = gamma / self.c / self.spacing
        lw = int(truncate * gamma_pixels + 0.5)
        
        kernel_x = np.arange(-lw, lw+1)
        kernel = self.lorentz_profile(kernel_x, 0, gamma_pixels)
        kernel /= np.sum(kernel)
        return kernel
    
    def voigt_kernel(self, fwhm, gamma, truncate=4.0):
        ''' Voigt kernel from the convolution of Gaussian and Lorentzian kernels
    
        Parameters
        ----------
        fwhm : float
            Full width at half maximum of the Gaussian kernel in km/s
        gamma : float
            Full width at half maximum of the Lorentzian kernel in km/s
        truncate : float
            Extent of the kernel as a multiple of gamma (default: 5)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''        
        _kernel_g = self.gaussian_kernel(fwhm, truncate)
        _kernel_l = self.lorentz_kernel(gamma, truncate)
        
        # note: here we use scipy.signal convolve instead of scipy.ndimage convolve1d
        # this way we use the strict definition of convolution (commutative)
        kernel = convolve(_kernel_g, _kernel_l, mode='same')
        kernel /= np.sum(kernel)
        return kernel

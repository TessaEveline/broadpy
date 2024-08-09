import numpy as np

def load_example_data(wave_range=(2320.0, 2330.0)):
    
    
    wave, flux = np.loadtxt('data/models_Teff_4300_logg_4.0_Z_+0.0.txt').T # T=4300K, logg=4.0, solar metallicity
    # select wavelength region in nm
    mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
    assert np.any(mask), 'No data in the selected wavelength range'
    wave = wave[mask]
    flux = flux[mask]
    return wave, flux
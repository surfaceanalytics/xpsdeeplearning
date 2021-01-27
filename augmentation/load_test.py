# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:45:09 2020

@author: pielsticker
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt


def _write_scatter_text(aug_values, index):
    scatterer = aug_values['scatterer'][index]
    distance = aug_values['distance'][index]
    pressure = aug_values['pressure'][index]
            
    scatterers = {'0' : 'He',
                  '1' : 'H2', 
                  '2' : 'N2',
                  '3' : 'O2'}       
    scatterer_name =  scatterers[str(scatterer[0])]

    name_text = 'Scatterer: ' + scatterer_name + ', '
        
    pressure_text = '{:.1f}'.format(float(pressure)) + ' mbar, '  
      
    distance_text = 'd = ' + \
                    '{:.1f}'.format(float(distance)) + ' mm'
                
    scatter_text = name_text + pressure_text + distance_text + '\n'
    
    return scatter_text


def _write_aug_text(aug_values, index):
    shift_x = aug_values['shiftx'][index]
    noise = aug_values['noise'][index]
    fwhm = aug_values['FWHM'][index]
            
            
    if (fwhm != None and fwhm != 0):
        fwhm_text = 'FHWM: ' + \
            str(np.round(float(fwhm), decimals = 2)) + ', '
    else:
        fwhm_text = 'FHWM: not changed' + ', '
                
    if (shift_x != None and shift_x != 0):            
        shift_text = ' Shift: ' + \
                    '{:.2f}'.format(float(shift_x)) + ', '
    else:
        shift_text = ' Shift: none' + ', '
                
    if (noise != None and noise != 0):
        noise_text = 'S/N: ' + '{:.1f}'.format(noise)     
    else:
        noise_text = 'S/N: not changed'
                
    aug_text = '\n' + fwhm_text + shift_text + noise_text + '\n'
        
    if 'scatterer' in aug_values.keys():
        aug_text += _write_scatter_text(aug_values, index)
    else:
        aug_text += 'Spectrum not scattered.'
    
    return aug_text
        
        
def plot_random(X, y, no_of_spectra, aug_values): 
    no_of_cols = 5
    no_of_rows = int(no_of_spectra/no_of_cols)
    if (no_of_spectra % no_of_cols) != 0:
        no_of_rows += 1
            
    fig, axs = plt.subplots(nrows = no_of_rows, ncols = no_of_cols)
    plt.subplots_adjust(left = 0.125, bottom = 0.5,
                        right=4.8, top = no_of_rows,
                        wspace = 0.2, hspace = 0.2)
    
    for i in range(no_of_spectra):
        energies = np.arange(694, 750.05, 0.05)

        intensities = X[i]
        label = str(np.round(y[i],2))
                                  
        row, col = int(i/no_of_cols), i % no_of_cols
        axs[row, col].plot(np.flip(energies),intensities)
        axs[row, col].invert_xaxis()
        axs[row, col].set_xlim(750.05,694)
        axs[row, col].set_xlabel('Binding energy (eV)')
        axs[row, col].set_ylabel('Intensity (arb. units)')   

        try:
            aug_text = _write_aug_text(aug_values,
                                       index = i)
            label += aug_text
        except:
            pass                     

        axs[row, col].text(0.025, 0.3, label,
                           horizontalalignment='left',
                           verticalalignment='top',
                           transform = axs[row, col].transAxes,
                           fontsize = 12) 


#output_file = r'U:\Simulations\20201118_iron_Mark_variable_linear_combination_gas_phase.h5'
output_file = r'C:\Users\pielsticker\Simulations\20210118_palladium_linear_combination_gas_phase.h5'
no_of_spectra = 20
vals =  {}
aug_values = {}
with h5py.File(output_file, 'r') as hf:
    keys = list(hf.keys())
    size = hf['X'].shape[0]
    r = np.random.randint(0, size)
    X = hf['X'][r:r+no_of_spectra,:,:]
    y = hf['y'][r:r+no_of_spectra,:]
    
    #names = hf['names'][:]
    
    for key in list(hf.keys()):
        if key not in ['X', 'y']:
            min_name = key + '_min'
            max_name = key + '_max'
            vals[min_name] = min(hf[key][:])
            vals[max_name] = max(hf[key][:])
            aug_values[key] = hf[key][r:r+no_of_spectra]
            
print(output_file)
plot_random(X, y, no_of_spectra, aug_values)
            

            
            
import numpy as np
import pywt
import mrcfile
import scipy.fftpack as fft
from scipy.ndimage import convolve
from math import sqrt

# Define the path to the input tomogram file
tomogram_path = "path/to/tomogram.mrc"
# Define the path to the input template file (in CIF format)
template_path = "path/to/template.cif"

# Load the tomogram data
with mrcfile.open(tomogram_path, permissive=True) as mrc:
    tomogram_data = np.array(mrc.data)

# Load the template data (as a density map)
with open(template_path, 'r') as f:
    template_data = []
    for line in f:
        if line.startswith("_atom_site"):
            x = float(line[80:88])
            y = float(line[88:96])
            z = float(line[96:104])
            density = float(line[60:67])
            template_data.append((x, y, z, density))
    # Convert template data to a 3D array (density map)
    template_data = np.array(template_data)
    template_data = template_data[:, :-1]
    template_shape = tuple(np.ceil(template_data.max(axis=0) - template_data.min(axis=0) + 1).astype(int))
    template_data -= template_data.min(axis=0)
    template_map = np.zeros(template_shape)
    template_map[template_data[:, 0].astype(int), template_data[:, 1].astype(int), template_data[:, 2].astype(int)] = template_data[:, 3]

# Define the WZM function
def wzm(image):
    # Define the Zernike polynomial order (N) and radius (R)
    N = 10
    R = min(image.shape) / 2.0
    # Define the WZM function
    wzm_func = lambda x: pywt.dwt2(x, 'haar')
    # Define the Zernike polynomials
    Z = np.zeros((N+1, image.shape[0], image.shape[1]))
    for n in range(N+1):
        for m in range(-n, n+1, 2):
            Rnm = sqrt(2*(n+1)) / (sqrt(np.pi) * R)
            Znm = Rnm * pywt.coeffs_to_array(pywt.wavedec2(pywt.pad(pywt.pad(pywt.pad(pywt.pad(np.zeros_like(image), [(0, image.shape[0] % 2), (0, image.shape[1] % 2)]), [(image.shape[0]//2, image.shape[0]//2), (image.shape[1]//2, image.shape[1]//2)]), 'symmetric'), [(1, 0), (1, 0)], mode='symmetric'), mode='symmetric', level=n, wavelet=wzm_func))
            if m < 0:
                Z[n] += np.real(Znm) * np.cos(m*np.angle(Znm)) - np.imag(Znm) * np.sin(m*np.angle(Znm))
            else:
                Z[n] += np.real(Znm) * np.cos(m*np.angle(Znm)) + np.imag(Znm) * np.sin(m*np.angle(Znm))
    # Calculate the WZM
    wzm = np.zeros(N+1)
    for n in range(N+1):
        wzm[n] = np.sum(image * Z[n])
    return wzm
  
  
''' 
OR:
  # Calculate the WZM for the tomogram and template
wzm_tomo = wzm(tomogram_data)
wzm_template = wzm(template_map)

# Normalize the template WZM
wzm_template /= np.sum(template_map)

# Calculate the cross-correlation between the tomogram and the template WZMs
cross_corr = fft.ifftshift(fft.ifftn(fft.fftn(wzm_tomo) * np.conj(fft.fftn(wzm_template))))
corr_max = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)

# Calculate the translation offset
offsets = np.array([corr_max[0] - (tomogram_data.shape[0] // 2), corr_max[1] - (tomogram_data.shape[1] // 2), 0])

# Output the offset
print("Translation offset (x, y, z):", offsets)
'''
  

# Calculate the WZM of the tomogram at each pixel
wzm_map = np.zeros((tomogram_data.shape[0]-template_shape[0], tomogram_data.shape[1]-template_shape[1], tomogram_data.shape[2]-template_shape[2], 11))
for i in range(wzm_map.shape[0]):
    for j in range(wzm_map.shape[1]):
        for k in range(wzm_map.shape[2]):
            image = tomogram_data[i:i+template_shape[0], j:j+template_shape[1], k:k+template_shape[2]]
            wzm_map[i, j, k] = wzm(image)

# Find the location of the template in the tomogram
corr_map = np.zeros((tomogram_data.shape[0]-template_shape[0], tomogram_data.shape[1]-template_shape[1], tomogram_data.shape[2]-template_shape[2]))
for i in range(corr_map.shape[0]):
    for j in range(corr_map.shape[1]):
        for k in range(corr_map.shape[2]):
            corr_map[i, j, k] = np.corrcoef(wzm_map[i, j, k], wzm(template_map))[0, 1]

max_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
print("Maximum correlation coefficient: {}".format(np.max(corr_map)))
print("Location of template in tomogram: ({}, {}, {})".format(max_idx[0], max_idx[1], max_idx[2]))
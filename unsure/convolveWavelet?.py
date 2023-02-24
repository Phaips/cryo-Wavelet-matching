import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import pywt
import mrcfile

# Load the tomogram data (as a density map)
with mrcfile.open('tomogram.mrc', permissive=True) as mrc:
    tomogram = mrc.data

# Load the template data (as a density map)
with mrcfile.open('template.cif.mrc', permissive=True) as mrc:
    template = mrc.data

# Compute the Wavelet Zernike Moment of the template
template_wzm = pywt.swt2(template, wavelet='db1', level=3)[0][0]

# Compute the WZM of the tomogram at each pixel
wzm_map = np.zeros_like(tomogram)
for i in range(tomogram.shape[0]):
    for j in range(tomogram.shape[1]):
        wzm_map[i, j] = pywt.swt2(tomogram[i:i+32, j:j+32], wavelet='db1', level=3)[0][0]

# Calculate the cross-correlation of the WZM maps
corr_map = convolve2d(wzm_map, np.flip(np.flip(template_wzm, 0), 1), mode='same')

# Find the location of the maximum correlation
max_pos = np.unravel_index(np.argmax(corr_map), corr_map.shape)

# Plot the tomogram and template
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(tomogram, cmap='gray')
ax[0].set_title('Tomogram')
ax[0].axis('off')
ax[1].imshow(template, cmap='gray')
ax[1].set_title('Template')
ax[1].axis('off')
plt.show()

# Plot the cross-correlation map and its maximum
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(corr_map, cmap='gray')
ax.plot(max_pos[1], max_pos[0], 'ro', markersize=10)
ax.set_title('Cross-correlation map')
ax.axis('off')
plt.show()

# Print the location of the maximum correlation
print('Max correlation at:', max_pos)

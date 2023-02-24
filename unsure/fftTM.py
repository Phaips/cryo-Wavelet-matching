import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks2d
from math import sqrt

# Load tomogram data
with mrcfile.open('tomogram.mrc', permissive=True) as mrc:
    tomogram = mrc.data.astype(np.float32)

# Load template data
with open('template.pdb', 'r') as f:
    lines = f.readlines()[10:]
    coords = []
    for line in lines:
        if line.startswith('ATOM'):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            density = float(line[54:60])
            coords.append((x, y, z, density))
    coords = np.array(coords)
    coords[:,:3] -= coords[:,:3].mean(axis=0) # center coordinates around origin
    coords[:,:3] += tomogram.shape[:3][::-1] // 2 # translate coordinates to center of tomogram
    template = np.zeros(tomogram.shape[:3], dtype=np.float32)
    for c in coords:
        x, y, z, d = c
        ix, iy, iz = np.round([x, y, z]).astype(np.int32)
        if 0 <= ix < template.shape[0] and 0 <= iy < template.shape[1] and 0 <= iz < template.shape[2]:
            template[ix, iy, iz] = d

# FFT tomo
wzm_tomogram = np.abs(np.fft.ifftn(np.fft.fftn(tomogram) / np.fft.fftn(np.abs(tomogram))))

# FTT template
wzm_template = np.abs(np.fft.ifftn(np.fft.fftn(template) / np.fft.fftn(np.abs(template))))

# Calculate cross-correlation map
corr_map = np.fft.ifftn(np.fft.fftn(wzm_tomogram) * np.fft.fftn(wzm_template).conj()).real

# Find peak in cross-correlation map
y_max, x_max = np.unravel_index(np.argmax(corr_map), corr_map.shape)

# plot tomogram and template
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(tomogram.max(axis=0), cmap='gray')
axes[0].set_title('Tomogram')
axes[1].imshow(template.max(axis=0), cmap='gray')
axes[1].set_title('Template')
plt.show()

# plot cross-correlation map
plt.imshow(corr_map, cmap='viridis')
plt.colorbar()
plt.title('Cross-Correlation Map')
plt.show()

# plot location of highest correlation
plt.imshow(tomogram.max(axis=0), cmap='gray')
plt.plot(y_max, x_max, 'r+', markersize=20, markeredgewidth=5)
plt.title('Location of highest correlation')
plt.show()

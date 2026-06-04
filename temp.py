import h5py
import numpy as np
H5_FILE = "MOST_M00_knee_patches_16_100.h5"
with h5py.File(H5_FILE, 'r') as hf:
    all_labels = [hf[k]['kl_grade'][0] for k in hf.keys() 
                  if 'kl_grade' in hf[k]]
    
labels = np.array(all_labels)
print("All unique labels in H5:", np.unique(labels))
print("Grade 2 count:", (labels == 2).sum())
print("Grade -999 count:", (labels == -999).sum())
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot

t1_fn = 'E:/Projects/Neuro-Diagnostic/ADNI/Annual 2 Yr 3T/002_S_0413/MPR____N3__Scaled/2006-05-19_16_17_47.0/S14782/ADNI_002_S_0413_MR_MPR____N3__Scaled_Br_20070216232854688_S14782_I40657.nii'
t2_fn = 'E:/Projects/Neuro-Diagnostic/ADNI/Annual 2 Yr 3T/002_S_0559/MPR____N3__Scaled/2006-06-27_18_28_33.0/S15922/ADNI_002_S_0559_MR_MPR____N3__Scaled_Br_20070319121214158_S15922_I45126.nii'
img = nib.load(t1_fn)
img2 = nib.load(t2_fn)

# print(img)

print("New")
print(img2)

img2_data = img2.get_fdata()


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


slice_0 = img2_data[141, :, :]
slice_1 = img2_data[:, 141, :]
slice_2 = img2_data[:, :, 141]
slices = [slice_0, slice_1, slice_2]
show_slices(slices)
plt.show()

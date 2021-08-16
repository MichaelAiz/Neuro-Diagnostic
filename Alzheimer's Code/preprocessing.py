# This file contains the main steps used to process the ADNI images and organize them into folders based on their labels

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import itk
import numpy as np
import os.path
import nibabel as nib
import matplotlib.pyplot




# image is resampled to a common isotropic resolution of 2mm^3
# remember to read in image using simpleitk
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0]):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)


    return resample.Execute(itk_image)

# convert a SimpleITK image to an ITK image via a numpy array, preserving metadata
def sitk_to_itk(sitk_image):
    itk_image = itk.GetImageFromArray(sitk.GetArrayFromImage(sitk_image), is_vector = sitk_image.GetNumberOfComponentsPerPixel()>1)
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())   
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(sitk_image.GetDirection()), [3]*2)))
    return itk_image

# convert an ITK image to a SimpleITK image via a numpy array, preserving metadata
def itk_to_sitk(itk_image):
    new_sitk_image = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_image), isVector=itk_image.GetNumberOfComponentsPerPixel()>1)
    new_sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    new_sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    new_sitk_image.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten())
    return new_sitk_image 

# register image to an atlas, MNI 305 atlas to be used

def register_image(sitk_fixed, sitk_moving):
    itk_fixed = sitk_to_itk(sitk_fixed)
    fixed_image = itk_fixed
    itk_moving = sitk_to_itk(sitk_moving)
    moving_image = itk_moving

    parameter_object = itk.ParameterObject.New()
    parameter_object = itk.ParameterObject.New()
    affine_parameter_map = parameter_object.GetDefaultParameterMap('affine') # rigid affine transformation is used, faster than non rigid bspline
    parameter_object.AddParameterMap(affine_parameter_map)
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.UpdateLargestPossibleRegion()
    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()





# This file contains the main steps used to process the ADNI images and organize them into folders based on their labels

from itk.support.extras import origin
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import itk
import numpy as np
import os.path
import nibabel as nib
import matplotlib.pyplot
import dltk.io.preprocessing
import dltk.io.augmentation
from nipype.interfaces.fsl import BET
from PIL import Image
from tensorflow.python.framework.type_spec import register
import constants

# image is resampled to a common isotropic resolution of 2mm^3
# remember to read in image using simpleitk
def resample_img(sitk_image, out_spacing=[2.0, 2.0, 2.0]):
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(sitk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)


    return resample.Execute(sitk_image)

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
# RETURNS AN ITK IMAGE
def register_img(sitk_fixed, sitk_moving):
    # convert sitk images to itk for registration using ITKElastix
    itk_fixed = sitk_to_itk(sitk_fixed)
    fixed_image = itk_fixed
    itk_moving = sitk_to_itk(sitk_moving)
    moving_image = itk_moving

    parameter_object = itk.ParameterObject.New()
    parameter_object = itk.ParameterObject.New()
    # set the parameter map to affine
    # parameter map defines how the transofrmation is performed
    affine_parameter_map = parameter_object.GetDefaultParameterMap('affine') # rigid affine transformation is used, faster than non rigid bspline
    parameter_object.AddParameterMap(affine_parameter_map)
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.UpdateLargestPossibleRegion()
    result_image = elastix_object.GetOutput()
    return result_image

def register_and_save(filename, path, atlas, description_csv):
    # # seperate filename by "_"
    split_name = filename.strip().split('_')
    # find and save image ID
    image_ID = split_name[-1][0:-4]
    label = ""
    # read the csv file containing label information as a pandas dataframe
    description = pd.read_csv(description_csv)
    IDs = description['Image Data ID']
    # search data frame for ID and extract corresponding label
    for row_index, id in enumerate(IDs):
        if(IDs.iloc[row_index] == image_ID):
            row = description.iloc[row_index]
            label = row['Group']
            print(f"{image_ID} is {label}") 
            break

    # resample the atlas
    sitk_atlas = sitk.ReadImage(atlas)
    resampled_sikt_atlas = resample_img(sitk_atlas)
    # resample the image
    complete_image_path = os.path.join(path, filename)
    sitk_moving = sitk.ReadImage(complete_image_path)
    resampled_sitk_moving = resample_img(sitk_moving)

    # register the image and save to registered database
    registered_image = register_img(resampled_sikt_atlas, resampled_sitk_moving)
    destination_path = os.path.join('E:/Projects/Neuro-Diagnostic/ADNI/Registered', label, filename)
    itk.imwrite(registered_image, destination_path)

# loop through original database and register all images, save in registered database
def register_images():
    description_csv = ''
    for subdir in constants.DB_SUBFOLDERS:
        for path, dirs, files in os.walk(constants.DATABASE + subdir):
            if files: 
                for file in files:
                    if(file.endswith('.csv')):
                        description_csv = os.path.join(path, file)
                    else:
                        try: 
                            register_and_save(file, path, constants.ATLAS, description_csv)
                        except RuntimeError:
                            print('Exception with', os.path.join(path, file))


# conversion algorithm consists of taking 16 horizontal sliced and placing them in a 4x4 grid
# slices are taken at levels that provide the most information about the brain 
def convert_to_2D(img):
    img_2D = np.empty((440, 344)) # 2D image shape is derived from the shape of the remaining planes of the 3D image * 4

    # set the cutting upper and lower limit, cuts will be between 30 and 60
    top = 60
    bottom = 30
    n_cuts = 16

    cut_iterator = top
    row_iterator = 0
    col_iterator = 0

    for cut_amount in range(n_cuts):
        cut = img[cut_iterator, :, :]
        cut_iterator -= 2

        # moves to the next column if the cut amount is 4, 8, 12, or 16
        if (cut_amount in [4, 8, 12]):
            row_iterator = 0
            col_iterator += cut.shape[1]

        # copy over the cut into the 2D image array, pixel by pixel
        for i in range(cut.shape[0]):
            for j in range(cut.shape[1]):
                img_2D[i + row_iterator, j + col_iterator] = cut[i, j]
        row_iterator += cut.shape[0]
    
    # image is changed to a 3-channel format since this is standard for most models
    return np.repeat(img_2D[None, ...], 3, axis = 0).T


# loads 2D image with the corresponding label, used for generation of datasets
def load_2D_with_label(img_path):
    label = constants.LABEL_MAP[(img_path.split('/')[-2])]
    sitk_img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(sitk_img)
    # image whitening is applied
    img = dltk.io.preprocessing.whitening(img)

    img = convert_to_2D(img)
    return img, label

# processes a single image to feed into a trained model
# takes in a nii file
def process_single_img(file):
    original_atlas = sitk.ReadImage(constants.ATLAS)
    resampled_atlas = resample_img(original_atlas)
    original_img = sitk.ReadImage(file)
    resampled_img = resample_img(original_img)
    registered_img = register_img(resampled_atlas, resampled_img)
    # image registration returns an itk image, so use itk methods
    img = itk.GetArrayFromImage(registered_img)
    img = dltk.io.preprocessing.whitening(img)
    image_2D = convert_to_2D(img)
    print(image_2D.shape)
    plt.imshow(image_2D)
    plt.show()
    # reshapes image for feeding into model
    image_2D = np.reshape(image_2D, (440, 344, 3))
    plt.imshow(image_2D)
    plt.show()
    image_2D = np.expand_dims(image_2D, axis=0)
    return image_2D

img = process_single_img('E:/Projects/Neuro-Diagnostic/ADNI\Original/Annual 2 Yr 3T/136_S_0426/MPR____N3__Scaled/2006-05-30_10_46_09.0/S15017/ADNI_136_S_0426_MR_MPR____N3__Scaled_Br_20070215213410384_S15017_I40378.nii')


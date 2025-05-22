#%%
import vtk
import numpy as np
from numpy.linalg import inv
import nibabel as nib

#from utils import create_affine_from_Eidolon
#%%

## Chargement de l`affine

#Method 1 : Unknown affine
# affine = nib.load("C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/20160906131917_AO_SINUS_STACK_CINES_29.nii").affine  ##retrieves the affine transformation matrix from the NIfTI file (image coord -> IRL coord)
#np.save("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/test_niftiheader_29.npy",affine)


#Method 2 : Known affine affine
affine = np.load("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/20160906131917_LVOT_SSFP_CINE_25.nii/test_niftiheader_LVOT_25.npy")

affine_list = affine.reshape(16).tolist()

def transformPolyData(polyData, transform):

    t = vtk.vtkTransformPolyDataFilter()
    t.SetTransform(transform)
    t.SetInputData(polyData)
    t.Update()
    return t.GetOutput()



#%%
tr = vtk.vtkTransform()
tr.SetMatrix(affine_list)

# affine3D = np.load('C:/codes/3Dmodelling/068 - Health control/niftiheader_3D.npy')
# test = inv(affine3D)
# affine_list3D = test.reshape(16).tolist()

# tr3D = vtk.vtkTransform()
# tr3D.SetMatrix(affine_list3D)

for i in range(40):   ##range value is equal to the number of time step, 40 in our case
    reader = vtk.vtkPolyDataReader()
    writer = vtk.vtkPolyDataWriter()

    reader.SetFileName("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/gradient_LVOT_25_{}.vtk".format(i))
    reader.Update()
    pd = reader.GetOutput()


    # Create a VTK affine transform
    

    out = transformPolyData(pd,tr)
    # test = transformPolyData(out,tr3D)

    writer.SetFileName("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/rotated_LVOT_25_{}.vtk".format(i))
    writer.SetInputData(out)
    writer.Write()
# %% 3D and STL


# affine3D = np.load('C:/codes/3Dmodelling/068 - Health control/niftiheader_3D.npy')
# affine_list3D = affine3D.reshape(16).tolist()

# # Load the STL file
# stl_reader = vtk.vtkSTLReader()
# stl_reader.SetFileName("C:/codes/3Dmodelling/068 - Health control/arch_clipped.stl")
# stl_reader.Update()
# stl_polydata = stl_reader.GetOutput()

# # Create a VTK affine transform
# tr3D = vtk.vtkTransform()
# tr3D.SetMatrix(affine_list3D)

# # Apply the affine transform to the STL polydata
# transformed_stl = transformPolyData(stl_polydata, tr3D)

# # Write the transformed STL polydata to a new file
# stl_writer = vtk.vtkSTLWriter()
# stl_writer.SetFileName("C:/codes/3Dmodelling/068 - Health control/transformed_model.stl")
# stl_writer.SetInputData(transformed_stl)
# stl_writer.Write()

# %%

# %% ###############################################################################################
# IMPORTS
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

import vtkmodules.all as vtk

from vtk import vtkPolyDataWriter

import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2

from vtkmodules.vtkCommonColor import vtkNamedColors

from vtkmodules.vtkCommonCore import (
    vtkFloatArray,
    vtkIdList,
    vtkPoints
)

from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData
)

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


# %% ###############################################################################################
# WRITING THE WHOLE SURFACE AS VTK

affine = nib.load("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/20160906131917_LVOT_SSFP_CINE_25.nii/20160906131917_LVOT_SSFP_CINE_25.nii").affine  ##retrieves the affine transformation matrix from the NIfTI file (image coord -> IRL coord)
mask = nib.load("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/20160906131917_LVOT_SSFP_CINE_25.nii/LVOT_SSFP_CINE_25_Segmentation.nii/LVOT_SSFP_CINE_25_Segmentation.nii").get_fdata()  ##Get_fdata transforms NifTy image data into a NumPy array
plt.imshow(mask[:,:,0,0])


np.save("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/20160906131917_LVOT_SSFP_CINE_25.nii/test_niftiheader_LVOT_25.npy",affine)

#%%

mask = nib.load("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/20160906131917_LVOT_SSFP_CINE_25.nii/LVOT_SSFP_CINE_25_Segmentation.nii/LVOT_SSFP_CINE_25_Segmentation.nii").get_fdata()  ##Get_fdata transforms NifTy image data into a NumPy array

# FIND NONZERO INDICES
for i in range(np.shape(mask)[3]):
    nonzero_indices = np.column_stack(np.nonzero(mask[:,:,:,i]))  ##trouve les indices des éléments non nuls dans la tranche courante. Et les empile ces indices en colonnes pour former un tableau 2D.

    # Convert to vtkPolyData
    poly_data = vtkPolyData()   ##Crée un objet vtkPolyData pour stocker les données de la surface
    points = vtkPoints()

    for j, (x,y,z) in enumerate(nonzero_indices):  ##Parcourt les indices non nuls et insère chaque point dans l'objet vtkPoints
        print(j,(x,y,z))
        points.InsertPoint(j,(x, y, z))

    poly_data.SetPoints(points)  ##Associe les points à l'objet vtkPolyData

# WRITING THE SURFACE AS VTK

    writer = vtkPolyDataWriter()   ##Crée un objet vtkPolyDataWriter pour écrire les données dans un fichier VTK
    writer.SetFileName("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/surface_LVOT_25_{}.vtk".format(i))   ##Définit le nom du fichier de sortie
    writer.SetInputData(poly_data)
    writer.Write()

# %% ###############################################################################################
# HULL METHOD FOR CONTOURS

# from scipy.spatial import ConvexHull
# from vtk import vtkPolyDataWriter

# for i in range(np.shape(mask)[3]):
#     reader = vtk.vtkPolyDataReader()
#     reader.SetFileName("C:/codes/3Dmodelling/Patient_743425/2018/2DstacksMRI/surface_11_{}.vtk".format(i))
#     reader.Update()
#     # Get the points from the VTK file
#     points = reader.GetOutput().GetPoints()
#     # Convert VTK points to a numpy array
#     points_array = np.array([points.GetPoint(j) for j in range(points.GetNumberOfPoints())])

#     # Find the convex hull
#     hull = ConvexHull(points_array[:, :2])  # Considering only x and y coordinates

#     # Plot the original points
#     # plt.plot(points_array[:, 0], points_array[:, 1], 'o')

#     hull_indices = np.unique(hull.simplices.flat)
#     hull_pts = points_array[hull_indices, :]

#     # Plot the  hull points
#     # plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'ro', alpha=.25, markersize=10)
#     # plt.xlabel('X')
#     # plt.ylabel('Y')
#     # plt.title('Convex Hull - Outer Contour')
#     # plt.show()

#     # Convert to vtkPolyData
#     poly_data = vtkPolyData()
#     points = vtkPoints()

#     for k, (x,y,z) in enumerate(points_array):
#         print(k,(x,y,z))
#         points.InsertPoint(k,(x, y, z))

#     poly_data.SetPoints(points)

#     writer = vtkPolyDataWriter()
#     writer.SetFileName("C:/codes/3Dmodelling/Patient_743425/2018/2DstacksMRI/hull{}.vtk".format(i))
#     writer.SetInputData(poly_data)
#     writer.Write()

# %% ###############################################################################################
# GRADIENT METHOD FOR CONTOURS

affine = nib.load("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/20160906131917_LVOT_SSFP_CINE_25.nii/20160906131917_LVOT_SSFP_CINE_25.nii").affine  ##retrieves the affine transformation matrix from the NIfTI file (image coord -> IRL coord)
mask = nib.load("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/20160906131917_LVOT_SSFP_CINE_25.nii/LVOT_SSFP_CINE_25_Segmentation.nii/LVOT_SSFP_CINE_25_Segmentation.nii").get_fdata()  ##Get_fdata transforms NifTy image data into a NumPy array

plt.imshow(mask[:,:,:,0])
print(np.shape(mask[:,:,:,0]))


#%%
for i in range(np.shape(mask)[3]):
    matrix = mask[:,:,:,i]

    gradient_x = np.gradient(matrix, axis=0)
    gradient_y = np.gradient(matrix, axis=1)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # plt.imshow(gradient_magnitude)

    gradient_mask = (gradient_magnitude > 0).astype(int)
    normalized_matrix = matrix * gradient_mask
    plt.imshow(normalized_matrix)
    plt.imshow(normalized_matrix, cmap='gray', vmin=0, vmax=1)
    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')

    plt.show()
    nonzero_indices = np.column_stack(np.nonzero(normalized_matrix))

    # Convert to vtkPolyData
    poly_data = vtkPolyData()
    points = vtkPoints()

    for j, (x,y,z) in enumerate(nonzero_indices):
        print(j,(x,y,z))
        points.InsertPoint(j,(x, y, z))

    poly_data.SetPoints(points)

    writer = vtkPolyDataWriter()
    writer.SetFileName("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/LVOT_view_test/Data/gradient_LVOT_25_{}.vtk".format(i))
    writer.SetInputData(poly_data)
    writer.Write()





# %%
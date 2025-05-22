#%% Import
import pyvista as pv
from vtkmodules.vtkCommonDataModel import vtkIterativeClosestPointTransform
from vtkmodules.vtkCommonDataModel import vtkPolyData
import numpy as np
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from scipy.spatial.transform import Rotation as R_scipy

#%% define path
path_source = "C:/Users/jr403s/Documents/Model_V2_1/Whole_heart_2016_42_mesh_V2_clipped.vtk"
path_target = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/gradient_29_test/gradient_29_16.vtk"

source = pv.read(path_source)
target = pv.read(path_target)

source_points = source.points  # numpy array de forme (N, 3)
target_points = target.points

# print("Source Points:", source_points)
# print("Target Points:", target_points)

#%% Convertir en UnstructuredGrid
source_vtk = source.cast_to_unstructured_grid()
target_vtk = target.cast_to_unstructured_grid()


#%% Apply ICP
icp = vtkIterativeClosestPointTransform()
icp.SetSource(source_vtk)
icp.SetTarget(target_vtk)
icp.GetLandmarkTransform().SetModeToRigidBody()
icp.StartByMatchingCentroidsOn()
icp.Update()
print("Matrice de transformation :")
print(icp.GetMatrix())
M_vtk = icp.GetMatrix()
M_transform = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        M_transform[i, j] = M_vtk.GetElement(i, j)

# Affichage
print("Matrice de transformation (format NumPy) :")
print(M_transform)

M_scaledRota=M_transform[:3, :3]
print("Matrice de rotation :")
print(M_scaledRota)
# %% Decomposing the scaled rotation matrix

# Extract scaling element
scale_x = np.linalg.norm(M_scaledRota[:, 0])
scale_y = np.linalg.norm(M_scaledRota[:, 1])
scale_z = np.linalg.norm(M_scaledRota[:, 2])

#Enlever les elements d`echelle pour isoler la rotation
R = np.zeros((3, 3))
R[:, 0] = M_scaledRota[:, 0]/scale_x
R[:, 1] = M_scaledRota[:, 1]/scale_y
R[:, 2] = M_scaledRota[:, 2]/scale_z

#Convertir la matrice de rotation en angles d’Euler
rot_obj = R_scipy.from_matrix(R)
euler_angles = rot_obj.as_euler('xyz', degrees=True)  

# Résultat :
angle_x, angle_y, angle_z = euler_angles

print("Angle_Euler :")
print(angle_x, angle_y, angle_z)
print("Mise_a_l_echelle :")
print(scale_x, scale_y, scale_z)
# %%

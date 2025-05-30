# %% ###############################################################################################
# IMPORTS
import numpy as np
import nibabel as nib
import itk
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt

from scipy.linalg import svd
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go

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


def read_points_vtk(file_vtk):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_vtk)
    reader.Update()
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    return points


#%% Load curve
# Script Python pour créer une load curve pour FEBio avec fonction personnalisée
def generate_load_curve(file_path, func, start=0, end=40, step=1):
    """
    file_path : chemin vers le fichier de sortie
    func      : fonction Python prenant x en argument, ex: lambda x: x + 1
    start     : valeur de départ pour x
    end       : valeur finale pour x
    step      : pas d'incrémentation
    """
    with open(file_path, 'w') as f:
        for x in range(start, end+1, step):
            y = func(x)
            f.write(f"{x} {y}\n")

#%%
import SimpleITK as sitk
import nibabel as nib
import numpy as np

# def apply_levelset_seg_simpleitk(nifti_image_path, nifti_mask_path, output_path=None, iterations=100):
#     """
#     Args:
#         nifti_image_path (str): Path to original image (.nii or .nii.gz)
#         nifti_mask_path (str): Path to the initial segmentation (binary) in NIfTI format
#         output_path (str, optional): Path to save the result (NIfTI). If None, do not save.
#         iterations (int): Number of iterations of the level-set evolution.

#     Returns:
#         SimpleITK.Image: The resulting segmented image (binary)
#     """

#     # Load data with SimpleITK
#     img_sitk = sitk.ReadImage(nifti_image_path, sitk.sitkFloat32)  # Convert to float32
#     mask_sitk = sitk.ReadImage(nifti_mask_path, sitk.sitkFloat32)  # Convert to float32

#     # Get the number of timesteps
#     num_timesteps = img_sitk.GetSize()[-1]

#     # Initialize a list to store the segmented timesteps
#     segmented_timesteps = []

#     # Iterate over each timestep
#     for t in range(num_timesteps):
#         # Extract the current timestep
#         img_timestep = img_sitk[:, :, :, t]
#         mask_timestep = mask_sitk[:, :, :, t]

#         # Iterate over each 2D slice in the timestep
#         num_slices = img_timestep.GetSize()[-1]
#         segmented_slices = []

#         for s in range(num_slices):
#             img_slice = img_timestep[:, :, s]
#             mask_slice = mask_timestep[:, :, s]

#             # Apply the GeodesicActiveContourLevelSet filter
#             filter = sitk.GeodesicActiveContourLevelSetImageFilter()
#             filter.SetNumberOfIterations(iterations)
#             filter.SetPropagationScaling(1.0)
#             filter.SetCurvatureScaling(1.0)
#             filter.SetAdvectionScaling(1.0)

#             result_slice = filter.Execute(mask_slice, img_slice)
#             segmented_slices.append(result_slice)

#         # Combine the segmented slices into a single timestep
#         segmented_timestep = sitk.JoinSeries(segmented_slices)
#         segmented_timesteps.append(segmented_timestep)

#     # Combine the segmented timesteps into a single 4D image
#     result_sitk = sitk.JoinSeries(segmented_timesteps)
#     # Binarise the level set result
#     binary_result = sitk.BinaryThreshold(result_sitk, lowerThreshold=-1e-6, upperThreshold=1e-6, insideValue=1, outsideValue=0)

#     # # Binarize each 3D volume separately
#     # binary_timesteps = [
#     #     sitk.BinaryThreshold(ts, lowerThreshold=-1e-6, upperThreshold=1e-6, insideValue=1, outsideValue=0)
#     #     for ts in segmented_timesteps
#     # ]

#     # # Combine into 4D image
#     # binary_result = sitk.JoinSeries(binary_timesteps)
#     # binary_result.SetOrigin(img_sitk.GetOrigin())
#     # binary_result.SetSpacing(img_sitk.GetSpacing())
#     # binary_result.SetDirection(img_sitk.GetDirection())

    
#     # Save the result if output_path is provided
#     if output_path:
#         sitk.WriteImage(binary_result, output_path)

#     return binary_result






def apply_levelset_seg_simpleitk(nifti_image_path, nifti_mask_path, output_path=None, iterations=100):
    """
    Args:
        nifti_image_path (str): Path to original image (.nii or .nii.gz)
        nifti_mask_path (str): Path to the initial segmentation (binary) in NIfTI format
        output_path (str, optional): Path to save the result (NIfTI). If None, do not save.
        iterations (int): Number of iterations of the level-set evolution.

    Returns:
        SimpleITK.Image: The resulting segmented image (binary)
    """

    # Load data with SimpleITK
    img_sitk = sitk.ReadImage(nifti_image_path, sitk.sitkFloat32)  # Convert to float32
    mask_sitk = sitk.ReadImage(nifti_mask_path, sitk.sitkFloat32)  # Convert to float32

    # Get the number of timesteps
    num_timesteps = img_sitk.GetSize()[-1]

    # Initialize a list to store the segmented timesteps
    segmented_timesteps = []

    # Iterate over each timestep
    for t in range(num_timesteps):
        # Extract the current timestep
        img_timestep = img_sitk[:, :, :, t]
        mask_timestep = mask_sitk[:, :, :, t]

        # Iterate over each 2D slice in the timestep
        num_slices = img_timestep.GetSize()[-1]
        segmented_slices = []

        for s in range(num_slices):
            img_slice = img_timestep[:, :, s]
            mask_slice = mask_timestep[:, :, s]

            # Apply the GeodesicActiveContourLevelSet filter
            filter = sitk.GeodesicActiveContourLevelSetImageFilter()
            filter.SetNumberOfIterations(iterations)
            filter.SetPropagationScaling(1.0)
            filter.SetCurvatureScaling(1.0)
            filter.SetAdvectionScaling(1.0)

            result_slice = filter.Execute(mask_slice, img_slice)
            segmented_slices.append(result_slice)

        # Combine the segmented slices into a single timestep
        segmented_timestep = sitk.JoinSeries(segmented_slices)
        segmented_timesteps.append(segmented_timestep)

    # Combine the segmented timesteps into a single 4D image
    result_sitk = sitk.JoinSeries(segmented_timesteps)

    # Save the result if output_path is provided
    if output_path:
        sitk.WriteImage(result_sitk, output_path)

    return result_sitk

# Test
affine = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/20160906131917_AO_SINUS_STACK_CINES_29.nii"
mask = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/20160906131917_AO_SINUS_STACK_CINES_29_Segmentation.nii"
output_path = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/20160906131917_AO_SINUS_STACK_CINES_29_Seg_levelset.nii"
seg = apply_levelset_seg_simpleitk(affine, mask, output_path, iterations=100)
print("Level set done")



#%%
# Saving the header of the segmentation (useful to rotate the model)
def Seg_header(file_path, seg_path, output_path):
    affine = nib.load(file_path).affine  ##retrieves the affine transformation matrix from the NIfTI file (image coord -> IRL coord)
    mask = nib.load(seg_path).get_fdata()  ##Get_fdata transforms NifTy image data into a NumPy array
    ##plt.imshow(mask[:,:,0,0])
    np.save(output_path,affine)



#%%
def Seg2Contours(nifti_path, segmentation_path, output_dir, affine_save_path):
    """
    nifti_path (str): Path to the original NIfTI file (to extract the affine matrix).
    segmentation_path (str): Path to the NIfTI file containing the 4D segmentation.
    output_dir (str): Output folder for generated .vtk files.
    affine_save_path (str): Path to save the affine matrix as .npy (same result as for the Seg_header function)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Chargement de la matrice affine et des données
    affine = nib.load(nifti_path).affine
    mask = nib.load(segmentation_path).get_fdata()
    np.save(affine_save_path, affine)

    ##print(f"Dimensions du masque : {mask.shape}")
    ##print("Traitement des volumes en cours...")

    for i in range(mask.shape[3]):
        matrix = mask[:, :, :, i]


        # Extraction des points

        nonzero_indices = np.column_stack(np.nonzero(matrix))

        poly_data_raw = vtkPolyData()
        points_raw = vtkPoints()

        for j, (x, y, z) in enumerate(nonzero_indices):
            points_raw.InsertPoint(j, (x, y, z))

        poly_data_raw.SetPoints(points_raw)

        vtk_path_raw = os.path.join(output_dir, f"surface_{i}.vtk")
        writer_raw = vtkPolyDataWriter()
        writer_raw.SetFileName(vtk_path_raw)
        writer_raw.SetInputData(poly_data_raw)
        writer_raw.Write()

        ##print(f"[{i}] Fichier brut écrit : {vtk_path_raw}")


        # Méthode Gradient pour extraire les contours

        gradient_x = np.gradient(matrix, axis=0)
        gradient_y = np.gradient(matrix, axis=1)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        gradient_mask = (gradient_magnitude > 0).astype(int)
        normalized_matrix = matrix * gradient_mask

        # plt.imshow(normalized_matrix[:, :, matrix.shape[2] // 2], cmap='gray', vmin=0, vmax=1)
        # plt.title(f'Gradient Slice - Volume {i}')
        # plt.xlabel('X Coordinate (pixels)')
        # plt.ylabel('Y Coordinate (pixels)')
        # plt.show()

        nonzero_indices_grad = np.column_stack(np.nonzero(normalized_matrix))

        poly_data_grad = vtkPolyData()
        points_grad = vtkPoints()

        for j, (x, y, z) in enumerate(nonzero_indices_grad):
            points_grad.InsertPoint(j, (x, y, z))

        poly_data_grad.SetPoints(points_grad)

        vtk_path_grad = os.path.join(output_dir, f"gradient_{i}.vtk")
        writer_grad = vtkPolyDataWriter()
        writer_grad.SetFileName(vtk_path_grad)
        writer_grad.SetInputData(poly_data_grad)
        writer_grad.Write()
        ##print(f"[{i}] Fichier gradient écrit : {vtk_path_grad}")


# Test
nifti_file = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/20160906131917_AO_SINUS_STACK_CINES_29.nii"
nifti_seg = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/20160906131917_AO_SINUS_STACK_CINES_29_Seg_levelset.nii"
output_path="C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/Test_levelset"
header_file="C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/Test_levelset/test_niftiheader_29_levelset.npy"
new_vtk = Seg2Contours(nifti_file, nifti_seg, output_path, header_file)
print("vtk conversion done")



#%% Fonction appliquant la transformee spatiale
def transformPolyData(polyData, transform):

    t = vtk.vtkTransformPolyDataFilter()
    t.SetTransform(transform)
    t.SetInputData(polyData)
    t.Update()
    return t.GetOutput()

#%%
def rotate_vtk(nifti_path, grad_path, output_path, A):
    """
    nifti_path (str): Path to the original NIfTI file (to extract the affine matrix).
    grad_path (str): Folder for existing gradient_{}.vtk files.
    output_path (str): Output folder for generated .vtk files.
    A (int): Number of timesteps in the model 
    """
    
    os.makedirs(output_path, exist_ok=True)

    # Load affine matrix from NIfTI file
    nifti_img = nib.load(nifti_path)
    affine = nifti_img.affine
    affine_list = affine.reshape(16).tolist()
    tr = vtk.vtkTransform()
    tr.SetMatrix(affine_list)

    for i in range(A):   ##range value is equal to the number of time step, A in our case
        reader = vtk.vtkPolyDataReader()
        writer = vtk.vtkPolyDataWriter()

        vtk_path_grad = os.path.join(grad_path, f"gradient_{i}.vtk")
        reader.SetFileName(vtk_path_grad)
        reader.Update()
        pd = reader.GetOutput()

        # Create a VTK affine transform
        out = transformPolyData(pd,tr)

        vtk_path_rota = os.path.join(output_path, f"rotated_{i}.vtk")
        writer.SetFileName(vtk_path_rota)
        writer.SetInputData(out)
        writer.Write()

# Test
nifti_file = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/20160906131917_AO_SINUS_STACK_CINES_29.nii"
grad_path="C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/Test_levelset"
output_path="C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/Test_levelset"

new_vtk = rotate_vtk(nifti_file, grad_path, output_path, 40)
print("vtk rotation done")

#%%
def vtk2Dslicer (model_path, output_path, A):
    """
    model_path (str): Input file for the existing 3D model.
    output_path (str): Output folder for generated .vtk files.
    A (int): Number of timesteps in the model
    """
    os.makedirs(output_path, exist_ok=True)


    for i in range(A):   ##range value is equal to the number of time step, 40 in our case
        #Reading the 3D model
        # if i<10 :
        #     vtk_path_model = os.path.join(model_path, f"Whole_heart_2016_42_mesh_V2_PostSim.t0{i}.vtk")
        # else :
        vtk_path_model = os.path.join(model_path, f"Whole_heart_2016_42_mesh_V2_PostSim.t{i:02d}.vtk")
        # print("reading : "+ vtk_path_model)

        # Lire le maillage 3D (Unstructured Grid)
        reader3D = vtk.vtkUnstructuredGridReader()
        reader3D.SetFileName(vtk_path_model)
        reader3D.Update()

        # Si vous avez besoin d’un PolyData pour la coupe :
        geometryFilter = vtk.vtkGeometryFilter()
        geometryFilter.SetInputData(reader3D.GetOutput())
        geometryFilter.Update()
        polydata3D = geometryFilter.GetOutput()

        #Reading the 2D MRI ##### Boucle for i in range(timestep) a implementer
        vtk_path_rota = os.path.join(output_path, f"rotated_29_{i}.vtk")
        reader2D = vtk.vtkPolyDataReader()
        reader2D.SetFileName(vtk_path_rota)
        reader2D.Update()
        polydata2D = reader2D.GetOutput()

        #Calcul du plan moyen et de sa normale
        # Récupère les points de la coupe
        points = polydata2D.GetPoints()
        n_points = points.GetNumberOfPoints()

        # Convertit en numpy pour faciliter le calcul
        pts = np.array([points.GetPoint(j) for j in range(n_points)])

        # Point moyen du plan
        center = np.mean(pts, axis=0)

        # Estimation de la normale via ACP (analyse en composantes principales)
        _, _, vh = np.linalg.svd(pts - center)
        normal = vh[2]  # La 3e composante (plus faible variance) correspond à la normale

        #Definition du plan de coupe dans vtk
        cutPlane = vtk.vtkPlane()
        cutPlane.SetOrigin(center)
        cutPlane.SetNormal(normal)

        #Application de la coupe au modele 3D
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(cutPlane)
        cutter.SetInputData(polydata3D)
        cutter.Update()
        cutPolyData = cutter.GetOutput()
        # print("Type de données sauvegardées :", cutPolyData.GetClassName())

        #Sauvegarde en vtk
        vtk_path_slice = os.path.join(output_path, f"transverse_slice_{i:03d}.vtk")
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(vtk_path_slice)
        writer.SetInputData(cutPolyData)
        writer.SetFileTypeToASCII()
        writer.Write()
    print("vtk2Dslicer - execution completed")


# # test
# path_3D = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/"
# path_out = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/"
# step = 40
# vtk2Dslicer(path_3D, path_out, step)


# %%
def shape_center(vtk_file):
    """
    vtk_file (str): Input file for the existing 3D model.
    A (int): Number of timesteps in the model
    """
    # reading the vtk file
    points = read_points_vtk(vtk_file)

    # Set the centre coordinates
    centre = [0.0, 0.0, 0.0]

    # Calculate the centre by averaging the coordinates of the points
    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)
        centre[0] += point[0]
        centre[1] += point[1]
        centre[2] += point[2]

    nmb_points = points.GetNumberOfPoints()
    centre[0] /= nmb_points
    centre[1] /= nmb_points
    centre[2] /= nmb_points
    # print(type(centre))
    # print(points.GetNumberOfPoints())
    return centre

# Example
# outputfile = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/"
# tab = np.empty((0, 3), int)
# for i in range(40) :
#     input_path = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/"
#     vtk_file = os.path.join(input_path, f"rotated_29_{i}.vtk")
#     centre = shape_center(vtk_file)
#     tab = np.vstack([tab, centre])
    # print("At step ", i, " Coordinates of the aorta centre :", centre)
# print("la matrice regroupant les centre est : ", tab)
# quatrieme_ligne = tab[3, :]
# print("4ème ligne de la matrice :", quatrieme_ligne)



# %% ################################################

def reorder_points(points):
    """
    This function is required because the slice generated link points by their z-value, thus creating a non-continuous path.

    The funtion reorders a list of 3D points to form a continuous path.
    This method follows a nearest neighbour algorithm:
    it starts from the first point and at each step adds the nearest point
    of those not yet used.
    
    Args:
        points (np.ndarray): array of shapes (N, 3), representing 3D points.
    """

    # Copy to avoid modifying the original data
    points = points.copy()

    # List for storing reordered points
    ordered = [points[0]]  # We start with the first point of the original list

    # Boolean table to see which points have already been used
    used = np.zeros(len(points), dtype=bool)
    used[0] = True  # the first point is already in use

    # Repeat until all points have been used
    for _ in range(1, len(points)):
        last_point = ordered[-1]  # last point added
        # Calculate the distance between the current point and the others
        dists = np.linalg.norm(points - last_point, axis=1)
        dists[used] = np.inf  # Skipping pint that are already used
        # Find the nearest unused point
        next_index = np.argmin(dists)
        # Add this point to the ordered list
        ordered.append(points[next_index])
        used[next_index] = True  

    return np.array(ordered)


#%%
def recentrer_points(file_vtk):
    """
    file_vtk (str) : Input file for the existing 3D model.
    """
    centre = shape_center(file_vtk)
    points_centres = []
    points=read_points_vtk(file_vtk)
    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)
        point_centre = [point[0] - centre[0], point[1] - centre[1], point[2] - centre[2]]
        points_centres.append(point_centre)
    return np.array(points_centres)

## Test
# input_path = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/rotated_29_2.vtk"
# matrice = recentrer_points(input_path)
# print(matrice)


#%% 
## Warning : The interpolator_points function tend to causes issue with complex 3D curves( Creating points in the volume but not along the curves)
def interpolator_points(points, target_point_number):
    """
    points (float) : An array of points to be interpolated.
    target_point_number : the desired number of points after interpolation.
    """
    # Create a linear interpolation for each dimension
    x = np.linspace(0, 1, points.shape[0])
    x_new = np.linspace(0, 1, target_point_number)

    # Interpolate each dimension
    interpolated_points = np.zeros((target_point_number, 3))
    for i in range(3):
        f = interp1d(x, points[:, i], kind='linear')
        interpolated_points[:, i] = f(x_new)
    ordered_interpolated_points=reorder_points(interpolated_points)
    return ordered_interpolated_points

# # Use Example
# new_number_points = min(matrice_A.shape[0], matrice_B.shape[0])
# matrice_A_interpolee = interpolator_points(matrice_A, new_number_points)
# matrice_B_interpolee = interpolator_points(matrice_B, new_number_points)


def interpolate_along_curve(points, target_point_number):
    # Calcul des longueurs cumulées (distance curviligne)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    cumulative_dist /= cumulative_dist[-1]  # normaliser entre 0 et 1

    # Nouvelles positions uniformément réparties
    target_distances = np.linspace(0, 1, target_point_number)

    # Interpolation le long de la courbe
    interpolated_points = np.zeros((target_point_number, 3))
    for i in range(3):
        f = interp1d(cumulative_dist, points[:, i], kind='linear')
        interpolated_points[:, i] = f(target_distances)
    ordered_interpolated_points=reorder_points(interpolated_points)
    return ordered_interpolated_points

#%%
######################################
####### Tentative optimisation #######
######################################

def kabsch_align(points_A, points_B):
    """
    points_A: np.ndarray of shape (N, 3) - reference point cloud
    points_B: np.ndarray of shape (N, 3) - point cloud to align
    """
    # Center both point clouds around their centroids
    A = points_A - points_A.mean(axis=0)
    B = points_B - points_B.mean(axis=0)

    # Compute covariance matrix
    H = B.T @ A

    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Compute the optimal rotation matrix
    R = Vt.T @ U.T

    # Reflection correction (ensure proper rotation with determinant +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # Rotate B and re-add A's centroid to complete alignment
    B_aligned = (B @ R) + points_A.mean(axis=0)

    # Optionally reorder points (Ensure a continuous curve)
    B_aligned=reorder_points(B_aligned)
    return B_aligned


def project_to_plane(points, normal, point_on_plane):
    """
    Projects 3D points onto a plane defined by a normal vector and a point on the plane.
    """
    normal = normal / np.linalg.norm(normal)
    vectors = points - point_on_plane
    distances = np.dot(vectors, normal)
    return points - np.outer(distances, normal)

def gap_calculator(points_A, points_B):
    gap = np.linalg.norm(points_A - points_B)
    return gap


##Test~~~~~

input_path_A = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/rotated_29_2.vtk"
matrice_A = recentrer_points(input_path_A)
matrice_A = reorder_points(matrice_A)
input_path_B = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/transverse_slice_002.vtk"
matrice_B = recentrer_points(input_path_B)
matrice_B = reorder_points(matrice_B)

# Interpolate the matrix
new_number_points = min(matrice_A.shape[0], matrice_B.shape[0])
matrice_A_interpolee = interpolate_along_curve(matrice_A, new_number_points)
matrice_B_interpolee = interpolate_along_curve(matrice_B, new_number_points)



############# Optimisation

# Calculate the original residual gap
gap_init = gap_calculator(matrice_A_interpolee, matrice_B_interpolee)
print("Gap before optimization :", gap_init)

matrice_A_opti = kabsch_align(matrice_B_interpolee, matrice_A_interpolee)

# # Reprojeter dans le plan de A (Ensure the optimised result is in the same plan as matrice_A)
# plane_normal = np.cross(matrice_A_interpolee[1] - matrice_A_interpolee[0], matrice_A_interpolee[2] - matrice_A_interpolee[0])
# plane_point = matrice_A_interpolee[0]
# matrice_B_opti = project_to_plane(matrice_B_opti, plane_normal, plane_point)

# Calculate the final residual gap
gap_final = gap_calculator(matrice_B_interpolee, matrice_A_opti)
print("Gap after optimization :", gap_final)



##### Visualization

# # Creating the figure and 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Draw the original contour (line + points)
# ax.plot(matrice_A_interpolee[:, 0], matrice_A_interpolee[:, 1], matrice_A_interpolee[:, 2], 'r-', alpha=0.6, label='Interpolation A')
# ax.scatter(matrice_A_interpolee[:, 0], matrice_A_interpolee[:, 1], matrice_A_interpolee[:, 2], c='r', s=20, alpha=0.5)

# # Draw interpolated points and their lines
# ax.plot(matrice_B_interpolee[:, 0], matrice_B_interpolee[:, 1], matrice_B_interpolee[:, 2],
#         'b--', label='Interpolation B')
# ax.scatter(matrice_B_interpolee[:, 0], matrice_B_interpolee[:, 1], matrice_B_interpolee[:, 2],
#            c='b', s=20)

# # Draw the optimised contour points and their lines
# ax.plot(matrice_B_opti[:, 0], matrice_B_opti[:, 1], matrice_B_opti[:, 2],
#         'g--', label='Optimised B')
# ax.scatter(matrice_B_opti[:, 0], matrice_B_opti[:, 1], matrice_B_opti[:, 2],
#            c='g', s=20)

# ax.set_title("Interpolation along the 3D contour")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.legend()
# plt.tight_layout()
# plt.show()

fig = go.Figure()

# Courbe A (rouge)
fig.add_trace(go.Scatter3d(
    x=matrice_A_interpolee[:, 0], y=matrice_A_interpolee[:, 1], z=matrice_A_interpolee[:, 2],
    mode='lines+markers',
    name='Interpolation A',
    line=dict(color='red'),
    marker=dict(size=3, color='red', opacity=0.5)
))

# Courbe B (bleu)
fig.add_trace(go.Scatter3d(
    x=matrice_B_interpolee[:, 0], y=matrice_B_interpolee[:, 1], z=matrice_B_interpolee[:, 2],
    mode='lines+markers',
    name='Interpolation B',
    line=dict(color='blue', dash='dash'),
    marker=dict(size=3, color='blue')
))

# Courbe B optimisée (vert)
fig.add_trace(go.Scatter3d(
    x=matrice_A_opti[:, 0], y=matrice_A_opti[:, 1], z=matrice_A_opti[:, 2],
    mode='lines+markers',
    name='Optimised A',
    line=dict(color='green', dash='dash'),
    marker=dict(size=3, color='green')
))

# Mise en page
fig.update_layout(
    title='Interpolation along the 3D contour',
    scene=dict(
        xaxis_title='X-Axis',
        yaxis_title='Y-Axis',
        zaxis_title='Z-Axis'
    ),
    legend=dict(x=0, y=1.5),
    margin=dict(l=0, r=0, b=0, t=30),
    height=400
)

fig.show()
# %%

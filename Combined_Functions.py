# %% ###############################################################################################
# IMPORTS
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

from scipy.linalg import svd

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

    affine = np.load(nifti_path)
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



#%%
def vtk2Dslicer (model_path, output_path, A):
    """
    model_path (str): Input file for the existing 3D model.
    output_path (str): Output folder for generated .vtk files.
    A (int): Number of timesteps in the model
    """
    os.makedirs(output_path, exist_ok=True)

    # #Reading the 3D model
    # reader3D = vtk.vtkPolyDataReader()
    # reader3D.SetFileName(model_path)
    # reader3D.Update()
    # polydata3D = reader3D.GetOutput()

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


# test
path_3D = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/"
path_out = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/"
step = 40
vtk2Dslicer(path_3D, path_out, step)


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
outputfile = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/"
tab = np.empty((0, 3), int)
for i in range(40) :
    input_path = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/"
    vtk_file = os.path.join(input_path, f"rotated_29_{i}.vtk")
    centre = shape_center(vtk_file)
    tab = np.vstack([tab, centre])
    # print("At step ", i, " Coordinates of the aorta centre :", centre)
# print("la matrice regroupant les centre est : ", tab)
# quatrieme_ligne = tab[3, :]
# print("4ème ligne de la matrice :", quatrieme_ligne)
# %%
######################################
####### Tentative optimisation #######
######################################

def centrer_points(file_vtk):
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
input_path = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/rotated_29_2.vtk"
matrice = centrer_points(input_path)
print(matrice)





#%%
def calculer_rotation_optimale(points_A, points_B):
    """
    points_A and points_B are NumPy arrays containing the coordinates of the points in the two sets of points to be aligned.
    """
    # Calculer la matrice de covariance
    H = np.dot(np.transpose(points_A), points_B)

    # Décomposition en valeurs singulières
    U, S, Vt = svd(H)

    # Calculer la matrice de rotation optimale
    R = np.dot(Vt.T, U.T)

    # Assurer que la matrice de rotation est une rotation propre (déterminant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R

def apply_rotation(points, R):
    points_rotates = np.dot(points, R.T)
    return points_rotates

def calculer_ecart_residuel(points_A, points_B):
    ecart = np.linalg.norm(points_A - points_B)
    return ecart


##Test~~~~~

input_path_A = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/rotated_29_2.vtk"
matrice_A = centrer_points(input_path_A)

input_path_B = "C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicerV2/transverse_slice_002.vtk"
matrice_B = centrer_points(input_path_B)

# # Calculer la rotation optimale
# R = calculer_rotation_optimale(matrice_A, matrice_B)

# # Appliquer la rotation à points_A
# points_A_rotates = apply_rotation(matrice_A, R)


# Calculer l'écart résiduel
ecart = calculer_ecart_residuel(matrice_A, matrice_B)
print("Écart résiduel après rotation :", ecart)

# %%

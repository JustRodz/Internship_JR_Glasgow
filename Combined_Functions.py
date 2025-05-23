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
    nifti_path (str): Chemin vers le fichier NIfTI original (pour extraire la matrice affine).
    segmentation_path (str): Chemin vers le fichier NIfTI contenant la segmentation 4D.
    output_dir (str): Dossier de sortie pour les fichiers .vtk générés.
    affine_save_path (str): Chemin pour enregistrer la matrice affine en .npy. (Meme resultat que pour la fonction Seg_header)
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
    nifti_path (str): Chemin vers le fichier NIfTI original (pour extraire la matrice affine).
    grad_path (str): Dossier d`entree pour les fichiers gradient_{}.vtk existants.
    output_path (str): Dossier de sortie pour les fichiers .vtk générés.
    A (int): Nombre de timestep du modele 
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
    model_path (str): Dossier d'entrée pour le modèle 3D existant.
    output_path (str): Dossier de sortie pour les fichiers .vtk générés.
    A (int): Nombre de timesteps du modèle.
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
        print("reading"+ vtk_path_model)

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
        print("Type de données sauvegardées :", cutPolyData.GetClassName())

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

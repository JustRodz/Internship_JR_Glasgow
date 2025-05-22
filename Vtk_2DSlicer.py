#%% Import

import vtk
import numpy as np
import vtk

#%%
#Reading the 3D model
reader3D = vtk.vtkPolyDataReader()
reader3D.SetFileName("C:/Users/jr403s/Documents/Model_V2_1/Whole_heart_2016_42_mesh_V2_clipped.vtk")
reader3D.Update()
polydata3D = reader3D.GetOutput()

#%%
for i in range(40):   ##range value is equal to the number of time step, 40 in our case
    #Reading the 2D MRI ##### Boucle for i in range(timestep) a implementer
    reader2D = vtk.vtkPolyDataReader()
    reader2D.SetFileName("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_rotatevtk/rotated_29_{}.vtk".format(i))
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

    #Sauvegarde en vtk
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/2DstacksMRI_29_test_2DSlicer/Tranverse_slice_29_{}.vtk".format(i))
    writer.SetInputData(cutPolyData)
    writer.Write()

# %%

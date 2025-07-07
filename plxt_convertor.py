import tecplot as tp
from vtk import vtkUnstructuredGrid, vtkPoints, vtkCellArray, vtkDoubleArray, vtkUnstructuredGridWriter

def convert_xplt_to_vtk(xplt_file_path, vtk_file_path):
    # Charger le fichier .xplt
    dataset = tp.data.load_tecplot(xplt_file_path)

    # Extraire les données
    x = dataset.variables['X'].values()
    y = dataset.variables['Y'].values()
    z = dataset.variables['Z'].values()
    # Supposons que vous avez une variable de champ, par exemple 'Pressure'
    pressure = dataset.variables['Pressure'].values()

    # Créer un objet VTK UnstructuredGrid
    ug = vtkUnstructuredGrid()
    points = vtkPoints()

    # Ajouter les points au grid
    for i in range(len(x)):
        points.InsertNextPoint(x[i], y[i], z[i])

    ug.SetPoints(points)

    # Ajouter les données de pression
    pressure_data = vtkDoubleArray()
    pressure_data.SetName("Pressure")
    for p in pressure:
        pressure_data.InsertNextValue(p)
    ug.GetPointData().SetScalars(pressure_data)

    # Écrire le fichier VTK
    writer = vtkUnstructuredGridWriter()
    writer.SetFileName(vtk_file_path)
    writer.SetInputData(ug)
    writer.Write()

# Exemple d'utilisation
xplt_file = "C:\Users\jr403s\Documents\Model_V2_1\Test_converter\Whole_heart_2016_42_mesh_V3_PreSim_0_0_0.xplt"
vtk_path = "C:\Users\jr403s\Documents\Model_V2_1\Test_converter\Whole_heart_2016_42_mesh_V3_PreSim_0_0_0.vtk"
convert_xplt_to_vtk(xplt_file, vtk_path)

# %% test 2 (using febio command line)

import os
import subprocess

def convert_xplt_to_vtk(febio_exe_path, xplt_file, output_dir, num_timesteps):
    """
    Convertit un fichier .xplt en plusieurs .vtk, un par timestep.
    
    Args:
        febio_exe_path (str): Chemin vers l'exécutable de FEBio Studio (CLI).
        xplt_file (str): Chemin vers le fichier .xplt.
        output_dir (str): Dossier de sortie pour les fichiers .vtk.
        num_timesteps (int): Nombre de time steps à exporter.
    """
    os.makedirs(output_dir, exist_ok=True)

    for t in range(num_timesteps):
        output_file = os.path.join(output_dir, f"frame_{t:03d}.vtk")

        command = [
            febio_exe_path,
            "--export", f"{xplt_file}",
            "--timestep", str(t),
            "--output", output_file
        ]
        print(f"Exporting timestep {t} to {output_file}")
        subprocess.run(command, check=True)

    print("Conversion terminée.")




# Exemple d'utilisation
febio_cli_path = r"C:/Program Files/FEBioStudio/bin/febio4.exe"  # À adapter selon ton installation
xplt_path = "C:/Users/jr403s/Documents/Model_V2_1/Test_converter/Whole_heart_2016_42_mesh_V3_PreSim_0_0_0.xplt"
vtk_path = "C:/Users/jr403s/Documents/Model_V2_1/Test_converter/Whole_heart_2016_42_mesh_V3_PreSim_0_0_0.vtk"
n_steps = 40
convert_xplt_to_vtk(febio_cli_path, xplt_path, vtk_path, n_steps)
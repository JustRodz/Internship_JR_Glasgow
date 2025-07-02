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
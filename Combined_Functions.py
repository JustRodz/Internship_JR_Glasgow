# %% ###############################################################################################
# IMPORTS
import numpy as np
import nibabel as nib
import itk
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import os
import cv2
import xml.etree.ElementTree as ET
import re

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

# Fonction appliquant la transformee spatiale
def transformPolyData(polyData, transform):

    t = vtk.vtkTransformPolyDataFilter()
    t.SetTransform(transform)
    t.SetInputData(polyData)
    t.Update()
    return t.GetOutput()



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





#%% Calculate the normal vector and acquire data on the BC

#Test to target the node_set related to precribed displacement B.C

def find_presc_disp_node(feb_file_path):
    with open(feb_file_path, 'r') as file:
        content = file.read()

    bc_pattern = re.compile(r'<bc name="(\w+)" node_set="@edge:(\w+)" type="prescribed displacement">')
    node_sets = bc_pattern.findall(content)
    return node_sets

def extract_node_coordinates(feb_file_path):
    with open(feb_file_path, 'r') as file:
        content = file.read()

    node_pattern = re.compile(r'<node id="(\d+)">([-\d.\s,]+)</node>')
    nodes = node_pattern.findall(content)

    node_coordinates = {}
    for node_id, coords in nodes:
        coords = np.array([float(c) for c in coords.split(',')])
        node_coordinates[int(node_id)] = coords

    return node_coordinates

### the input correspond with the node_coordinates shape
def calculate_svd_normal(points):
    """
    points: np.ndarray of shape (N, 3) - reference point cloud
    """
    # Center both point cloud around their centroids
    centered_points = points - np.mean(points, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)
    # print(cov_matrix)
    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(cov_matrix)

    # normal/direction vector
    normal_vector= Vt[-1,:]
    normal_vector = normal_vector/np.linalg.norm(normal_vector)
    
    return normal_vector     


# Creating the coordinate system associated with
def create_coordinate_system(center, normal_vector, points):

    # Normalize the normal vector to get the Z-axis
    z_axis = normal_vector / np.linalg.norm(normal_vector)

    # Calculate a temporary X-axis vector
    first_point = points[0]
    temp_x_axis = first_point - center

    # Normalize the temporary X-axis vector to get the X-axis
    x_axis = temp_x_axis / np.linalg.norm(temp_x_axis)

    # Calculate the Y-axis using the cross product of Z and X axes
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Ensure the X-axis is orthogonal to both Z and Y axes
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    return np.array([x_axis, y_axis, z_axis])


# Extract the data from the nodeset associated with the B.C
def extract_nodeset_data(feb_file_path, node_sets):
    node_coordinates = extract_node_coordinates(feb_file_path)

    with open(feb_file_path, 'r') as file:
        data = file.read()

    
    bc_data = []
    for bc_name, node_set_name in node_sets:
        # Locating the Edges markers
        edge_pattern = re.compile(rf'<Edge name="{node_set_name}">(.*?)</Edge>', re.DOTALL)
        edge_match = edge_pattern.search(data)

        node_data = []
        if edge_match:
            edge_content = edge_match.group(1)

            # Finding nodes ids
            line_pattern = re.compile(r'<line2 id="(\d+)">(\d+),(\d+)</line2>')
            line_matches = line_pattern.findall(edge_content)

            # Adding nodes ids
            for match in line_matches:
                line_id, node1, node2 = match
                node_data.append(int(node1))
                node_data.append(int(node2))

        node_data = list(set(node_data))  # Suppress duplicates    

        #Obtain coords
        points = np.array([node_coordinates[node_id] for node_id in node_data])
        #calculate the center
        center =np.mean(points, axis=0)
        #Calculate the normal vector
        normal_vector = calculate_svd_normal(points)
        #Calculate the coordinate system associated with the B.C
        coordinate_system = create_coordinate_system(center, normal_vector, points)
        # print("Coordinate System:\n", coordinate_system)
        bc_data.append([bc_name, normal_vector, center, coordinate_system])

    return bc_data

# Rotation matrix
def calc_rota_matrix(primary_coord_system, second_coord_system):
    R= np.dot(second_coord_system, primary_coord_system.T)
    return(R)


input_file_path = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_PreSim.feb"
node_of_interest=find_presc_disp_node(input_file_path)
nodes_data= extract_nodeset_data(input_file_path, node_of_interest)
node_coordinates = extract_node_coordinates(input_file_path)
print("Coordinates :")
print(node_of_interest)
print("Finished the search")
print(nodes_data[0][3])
print("Finished extracting node`s data")
original_coord_system = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) #We theorise that the feb file coordinate system is the orthonormal direct system
BC_coord_system = nodes_data[0][3]
transform_matrix = calc_rota_matrix(BC_coord_system, original_coord_system)
print(transform_matrix)


#%% Useful to modify the values of a specified B.C
def update_BC_parameters(febio_file, boundary_name, updates, output_file_path=None):
    # Load the XML file
    tree = ET.parse(febio_file)
    root = tree.getroot()

    # Iterate through all boundary conditions
    for bc in root.iter('bc'):
        if bc.attrib.get('type') == 'prescribed displacement' and bc.attrib['name'] == boundary_name:
            # Update the 'lc' attribute in the 'value' element (associated load curve)
            value_element = bc.find('value')
            if value_element is not None and 'lc' in updates:
                value_element.set('lc', str(updates['lc']))

            # Update the 'initial_value' attribute in the 'value' element (initial value of the prescribed element)
            value_element = bc.find('value')
            if value_element is not None and 'initial_value' in updates:
                value_element.text = str(updates['initial_value'])

            # Update the 'relative' element
            relative_element = bc.find('relative')
            if relative_element is not None and 'relative' in updates:
                relative_element.text = str(updates['relative'])

    # Determine the file path to write the updated XML
    output_path = output_file_path if output_file_path else febio_file

    # Write the updated XML back to the file
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        tree.write(f, encoding='unicode')

updates = {'initial_value': 0, 'relative': 0}
file_path = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_PreSim_modifier_test.feb"
update_BC_parameters(file_path, "PrescribedDisplacement2", updates)

#%%#%% Useful to modify the type of output file
def update_output(febio_file, updates, output_file_path=None):
    # Load the XML file
    tree = ET.parse(febio_file)
    root = tree.getroot()

    # Iterate through all boundary conditions
    for plotfile in root.iter('plotfile'):
        plotfile.set('type', str(updates['new_type']))


    # Determine the file path to write the updated XML
    output_path = output_file_path if output_file_path else febio_file

    # Write the updated XML back to the file
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        tree.write(f, encoding='unicode')

updates = {'new_type': 'vtk'}
input_dir = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation"
for i in range (0,6):
    for j in range (0, 6):
        for k in range (0, 11):
            file_path = os.path.join(input_dir, f"Whole_heart_2016_42_mesh_V3_PreSim_{i}_{j}_{k}.feb")
            update_output(file_path, updates)

#%% Script to test Update_BC_parameters

input_file_path = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_PreSim.feb"
node_of_interest=find_presc_disp_node(input_file_path)
nodes_data= extract_nodeset_data(input_file_path, node_of_interest)
node_coordinates = extract_node_coordinates(input_file_path)
original_coord_system = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) #We theorise that the feb file coordinate system is the orthonormal direct system
BC_coord_system = nodes_data[0][3]
R = calc_rota_matrix(BC_coord_system, original_coord_system)

file_path = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Whole_heart_2016_42_mesh_V3_PreSim_modifier_test.feb"
output_dir= "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation" 
for i in range (0,6):
    for j in range (0, 6):
        for k in range (0, 11):
            coords =np.array([i, j, k])
            coords_xyz = np.dot(R, coords) 
            output = os.path.join(output_dir, f"Whole_heart_2016_42_mesh_V3_PreSim_{i}_{j}_{k}.feb")
            x_bc = {'initial_value': coords_xyz[0]}
            update_BC_parameters(file_path, "PrescribedDisplacement2", x_bc, output)
            y_bc = {'initial_value': coords_xyz[1]}
            update_BC_parameters(output, "PrescribedDisplacement3", y_bc)
            z_bc = {'initial_value': coords_xyz[2]}
            update_BC_parameters(output, "PrescribedDisplacement4", z_bc)



#%%
# Saving the header of the segmentation (useful to rotate the model)
def Seg_header(file_path, seg_path, output_path):
    affine = nib.load(file_path).affine  ##retrieves the affine transformation matrix from the NIfTI file (image coord -> IRL coord)
    mask = nib.load(seg_path).get_fdata()  ##Get_fdata transforms NifTy image data into a NumPy array
    ##plt.imshow(mask[:,:,0,0])
    np.save(output_path,affine)

#%% LVOT
def LvotSeg2vtk(nifti_path, segmentation_path, output_dir):
    affine = nib.load(nifti_path).affine  ##retrieves the affine transformation matrix from the NIfTI file (image coord -> IRL coord)
    mask = nib.load(segmentation_path).get_fdata()
    # Plot each label in the mask, with larger images
    num_labels = int(mask.max())
    for label in range(1, num_labels + 1):
        for i in range(mask.shape[3]):
            label_mask = (mask[:, :, :, i] == label)
            matrix = mask[:, :, :, i]
            # plt.figure(figsize=(4, 4))
            # plt.imshow(label_mask, cmap='gray')
            # plt.title(f'Label {label}', fontsize=20)
            # plt.axis('off')
            # plt.show()

            # Extraction des points

            nonzero_indices = np.column_stack(np.nonzero(label_mask))

            poly_data_raw = vtkPolyData()
            points_raw = vtkPoints()

            for j, (x, y, z) in enumerate(nonzero_indices):
                points_raw.InsertPoint(j, (x, y, z))

            poly_data_raw.SetPoints(points_raw)

            vtk_path_raw = os.path.join(output_dir, f"surface_{label}_time_{i}.vtk")
            writer_raw = vtkPolyDataWriter()
            writer_raw.SetFileName(vtk_path_raw)
            writer_raw.SetInputData(poly_data_raw)
            writer_raw.Write()


affine = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_24.nii/20160906131917_LVOT_SSFP_CINE_24.nii"
mask = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_24.nii/LVOT_SSFP_CINE_24_Segmentation_LVOT.nii"
output_dir = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_24.nii/LVOT_SSFP_CINE_24_vtk"
LVOT_vtk_conversion = LvotSeg2vtk(affine, mask, output_dir)
print("LVOT segmentation converted to vtk")


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
    return interpolated_points

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
    return interpolated_points

#%% Other Strategy to extract the contour of a segmentation using the Gradient methodology. Main issue = Point are ordered based on their z coordinate (noncontinuous curve) + rougher shape
#Don`t use for LVOT seg where we omly export key points and a plain shape  
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

        vtk_path_grad = os.path.join(output_dir, f"contour_{i}.vtk")
        writer_grad = vtkPolyDataWriter()
        writer_grad.SetFileName(vtk_path_grad)
        writer_grad.SetInputData(poly_data_grad)
        writer_grad.Write()
        ##print(f"[{i}] Fichier gradient écrit : {vtk_path_grad}")


# Test
nifti_file = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/20160906131917_AO_SINUS_STACK_CINES_29.nii"
nifti_seg = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/20160906131917_AO_SINUS_STACK_CINES_29_Segmentation.nii"
output_path="C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/"
header_file="C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/test_niftiheader_29.npy"
new_vtk = Seg2Contours(nifti_file, nifti_seg, output_path, header_file)
print("vtk conversion done")




#%%
def rotate_vtk(nifti_path, grad_path, output_path, A):
    """
    nifti_path (str): Path to the original NIfTI file (to extract the affine matrix).
    grad_path (str): Folder for existing contour_{}.vtk files.
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

        vtk_path_grad = os.path.join(grad_path, f"contour_{i}.vtk")
        reader.SetFileName(vtk_path_grad)
        reader.Update()
        pd = reader.GetOutput()

        # Create a VTK affine transform
        out = transformPolyData(pd,tr)

        vtk_path_rota = os.path.join(output_path, f"rotated_{i}.vtk")
        writer.SetFileName(vtk_path_rota)
        writer.SetInputData(out)
        writer.Write()

# # Test
# nifti_file = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/20160906131917_AO_SINUS_STACK_CINES_29.nii"
# grad_path="C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/"
# output_path="C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/"
# new_vtk = rotate_vtk(nifti_file, grad_path, output_path, 40)
# print("vtk rotation done")


def rotateLVOT(nifti_path, grad_path, output_path, A, B):
    """
    nifti_path (str): Path to the original NIfTI file (to extract the affine matrix).
    grad_path (str): Folder for existing contour_{}.vtk files.
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
        for j in range(1, B+1):   ##range value is equal to the number associated with the key point
            reader = vtk.vtkPolyDataReader()
            writer = vtk.vtkPolyDataWriter()

            vtk_path_grad = os.path.join(grad_path, f"surface_{j}_time_{i}.vtk")
            reader.SetFileName(vtk_path_grad)
            reader.Update()
            pd = reader.GetOutput()

            # Create a VTK affine transform
            out = transformPolyData(pd,tr)

            vtk_path_rota = os.path.join(output_path, f"rotated_surface_{j}_time_{i:02d}.vtk")
            writer.SetFileName(vtk_path_rota)
            writer.SetInputData(out)
            writer.Write()


# Test
nifti_file = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_24.nii/20160906131917_LVOT_SSFP_CINE_24.nii"
grad_path = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_24.nii/LVOT_SSFP_CINE_24_vtk"
output_path = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_24.nii/LVOT_SSFP_CINE_24_vtk"
new_vtk = rotateLVOT(nifti_file, grad_path, output_path, 40, 7)
print("vtk rotation done")



#%%
import os
import vtk
import numpy as np




def vtk2Dslicer (model_path, input_seg_path, output_path, model3D_name_pattern, seg_name_pattern, output_pattern, A):
    """
    model_path (str): Input file for the existing 3D model.
    output_path (str): Output folder for generated .vtk files.
    A (int): Number of timesteps in the model
    model3D_name_pattern (str): Pattern for the 3D model filenames, using {i} for the timestep.
    seg_name_pattern (str): Pattern for the rotated segmentation filenames, using {i} for the timestep.
    """
    os.makedirs(output_path, exist_ok=True)


    for t in range(A):   ##range value is equal to the number of time step, 40 in our case
        vtk_path_model = os.path.join(model_path, model3D_name_pattern.format(t=t))
        print(f"Reading model: {vtk_path_model}")  # Debug print
        if not os.path.exists(vtk_path_model):
            print(f"File not found: {vtk_path_model}")
            continue

        # Reading the 3D mesh (Unstructured Grid)
        reader3D = vtk.vtkUnstructuredGridReader()
        reader3D.SetFileName(vtk_path_model)
        reader3D.Update()


        geometryFilter = vtk.vtkGeometryFilter()
        geometryFilter.SetInputData(reader3D.GetOutput())
        geometryFilter.Update()
        polydata3D = geometryFilter.GetOutput()

        #Reading the 2D MRI ##### Boucle for i in range(timestep) a implementer
        vtk_path_rota = os.path.join(input_seg_path, seg_name_pattern.format(t=t))
        # print(f"Reading segmentation: {vtk_path_rota}")  # Debug print
        #     if not os.path.exists(vtk_path_rota):
        #         print(f"File not found: {vtk_path_rota}")
        #         continue

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
        vtk_path_slice = os.path.join(output_path, output_pattern.format(t=t))
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(vtk_path_slice)
        writer.SetInputData(cutPolyData)
        writer.SetFileTypeToASCII()
        writer.Write()
    print("vtk2Dslicer - execution completed")
# except Exception as e:
#         print(f"An error occurred: {e}")


# # # test
# output_dir = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk"
# input_seg = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk"
# path_3D = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs"
# # output_path="C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_34/AO_SINUS_STACK_CINES_34_vtk/"
# file3D_pattern = "Whole_heart_2016_42_mesh_V3_PreSim.t{i:02d}.vtk"
# seg_pattern = "rotated_{i}.vtk"
# step = 40
# vtk2Dslicer(path_3D, input_seg, output_dir, file3D_pattern, seg_pattern, step)
# print("Model sliced")


#%% Associating key points from the MRI 2D segmentation to the 3D model

def find_closest_point(aorta_points, second_point):
    distances = np.linalg.norm(aorta_points - second_point, axis=1)
    closest_point_index = np.argmin(distances)
    return (aorta_points[closest_point_index], closest_point_index)


def write_vtk_points(file_path, points):
    points_vtk = vtk.vtkPoints()
    for point in points:
        points_vtk.InsertNextPoint(point)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points_vtk)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()

# Use case
aorta_file_pattern = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25.nii/LVOT_SSFP_CINE_25_vtk/transverse_slice_{i:03d}.vtk"
segment_file_pattern = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25.nii/LVOT_SSFP_CINE_25_vtk/rotated_surface_{j}_time_00.vtk"
output_file_pattern = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25.nii/LVOT_SSFP_CINE_25_vtk/closest_points_{j:02d}_pts_{i:02d}.vtk"

# Initialisation des listes pour stocker les distances
height = []
distances_bottom = []
distances_mid = []
distances_top = []

# Dictionnaire pour stocker les points de chaque surface à chaque timestep
surface_points = {j: [] for j in range(1, 8)}

for i in range(40):
    for j in range(1, 8):
        aorta_file = aorta_file_pattern.format(i=i)
        segment_file = segment_file_pattern.format(j=j)

        aorta_points = read_points_vtk(aorta_file)
        aorta_pts_array = np.array([aorta_points.GetPoint(k) for k in range(aorta_points.GetNumberOfPoints())])
        segment_points = read_points_vtk(segment_file)
        segment_pts_array = np.array([segment_points.GetPoint(k) for k in range(segment_points.GetNumberOfPoints())])

        if i == 0:
            # Trouver le point le plus proche pour le premier timestep
            closest_point, closest_index = find_closest_point(aorta_pts_array, np.mean(segment_pts_array, axis=0))
        else:
            # Trouver le point le plus proche du point précédent
            closest_point, closest_index = find_closest_point(aorta_pts_array, surface_points[j][-1])

        # Stocker le point pour la surface actuelle et le timestep actuel
        surface_points[j].append(closest_point)

        # Écrire le point le plus proche
        output_file = output_file_pattern.format(j=j, i=i)
        write_vtk_points(output_file, [closest_point])

    # Calculer et stocker les distances entre les paires de points spécifiées pour le timestep actuel
    if i < len(surface_points[1]):  # Assurez-vous que les points existent pour ce timestep
        height.append(np.linalg.norm(surface_points[1][i] - surface_points[2][i]))
        distances_bottom.append(np.linalg.norm(surface_points[2][i] - surface_points[3][i]))
        distances_mid.append(np.linalg.norm(surface_points[4][i] - surface_points[5][i]))
        distances_top.append(np.linalg.norm(surface_points[6][i] - surface_points[7][i]))




# Affichage des distances calculées
print("Distances entre les points 1 et 2 :", height)
print("Distances entre les points 2 et 3 :", distances_bottom)
print("Distances entre les points 4 et 5 :", distances_mid)
print("Distances entre les points 6 et 7 :", distances_top)


#%%
import vtk
import numpy as np

def extract_endpoints(filename):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    # get data and lines
    data = reader.GetOutput()
    lines = data.GetLines()

    # Initialise the liste
    endpoints = []

    # finding extremities
    lines.InitTraversal()
    idList = vtk.vtkIdList()
    while lines.GetNextCell(idList):
        if idList.GetNumberOfIds() > 0:
            # adding first and last point to the line
            endpoints.append(idList.GetId(0))
            endpoints.append(idList.GetId(idList.GetNumberOfIds() - 1))

    # Extract coordinates
    points = data.GetPoints()
    endpoint_coordinates = [points.GetPoint(i) for i in endpoints]

    return endpoint_coordinates

# Use case
filename = 'votre_fichier.vtk'
endpoints = extract_endpoints(filename)
print("Points des extrémités :", endpoints)
# %% Useful to obtain the coordinate of the center of a 2D shape
def shape_center(vtk_file):
    """
    vtk_file (str): Input file for the existing 3D model.
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
    # centre = [coord / nmb_points for coord in centre]
    centre[0] /= nmb_points
    centre[1] /= nmb_points
    centre[2] /= nmb_points

    return centre

def calculate_distances(tab_2DMRI, tab_3Dslice):
    """
    Calculate the Euclidean distances between corresponding centers in tab_2DMRI and tab_3Dslice.
    """
    distances = []
    for centre_2D, centre_3D in zip(tab_2DMRI, tab_3Dslice):
        distance = np.linalg.norm(np.array(centre_2D) - np.array(centre_3D))
        distances.append(distance)
    return distances


def gap_calculator_array(points_A, points_B):
    gap = np.linalg.norm(points_A - points_B)
    return gap

def gap_calculator_vtk(points_A, points_B):
    # Extraire les coordonnées des points
    coords_A = np.array([points_A.GetPoint(i) for i in range(points_A.GetNumberOfPoints())])
    coords_B = np.array([points_B.GetPoint(i) for i in range(points_B.GetNumberOfPoints())])

    # Interpolate the matrix
    new_number_points = min(coords_A.shape[0], coords_B.shape[0])
    if new_number_points == coords_A.shape[0] :
        coords_B = interpolate_along_curve(coords_B, new_number_points)
    else :
        coords_A = interpolate_along_curve(coords_A, new_number_points)
    # Calculer la différence et la norme
    gap = np.linalg.norm(coords_A - coords_B)
    return gap


# Example
input_path_2DMRI="C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/"
input_path_3Dslice="C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/"
tab_2DMRI = np.empty((0, 3), int)
tab_3Dslice = np.empty((0, 3), int)
tab_gap= np.empty((0, 3), int)
for i in range(40) :
    # Calculate &stock the coordinates of the center of the 2D MRI shape 
    vtk_file_2DMRI = os.path.join(input_path_2DMRI, f"rotated_{i}.vtk")
    centre_2DMRI = shape_center(vtk_file_2DMRI)
    tab_2DMRI = np.vstack([tab_2DMRI, centre_2DMRI])
    # Calculate &stock the coordinates of the center of the 3D model slice
    vtk_file_3Dslice = os.path.join(input_path_3Dslice, f"transverse_slice_{i:03d}.vtk")
    centre_3Dslice = shape_center(vtk_file_3Dslice)
    tab_3Dslice = np.vstack([tab_3Dslice, centre_3Dslice])
    # print("At step ", i, " Coordinates of the aorta centre :", centre)
    points_2DMRI=read_points_vtk(vtk_file_2DMRI)
    points_3Dslice=read_points_vtk(vtk_file_3Dslice)
    gap_2DMRI_3D = gap_calculator_vtk(points_2DMRI, points_3Dslice)
    print(f"Gap value at Timestep {i}: {gap_2DMRI_3D}")
    tab_gap= np.vstack([tab_gap, centre_3Dslice])

# print("la matrice regroupant les centres 2D MRI est : ", tab_2DMRI)
# print("la matrice regroupant les centres 3D model est : ", tab_3Dslice)

# Calculate distances between centers for each timestep
distances = calculate_distances(tab_2DMRI, tab_3Dslice)

# Print the distances
for i, distance in enumerate(distances):
    print(f"Distance at timestep {i}: {distance}")


#%% Simulation runner
import subprocess
import os

##set the folder where you stock the .feb file as base path
basepath = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation"

#adjust the ranges to coorespond with the simu
for i in range(0,11):
    for j in range(6):
        for k in range(6):
            #Specify input and output
            feb_path = os.path.join(basepath, f"Whole_heart_2016_42_mesh_V3_PreSim_{k}_{j}_{i}.feb")
            vtk_path = os.path.join(basepath, "jobs", f"Output_WH_{k}_{j}_{i}.vtk")

            # write the command line and run it
            cmd = ["febio4", "run", "-i", feb_path, "-p", vtk_path]
            print("Running:", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)

            #error case
            if result.returncode != 0:
                print(f"Error at index {i}:\n{result.stderr}")
            else:
                print(result.stdout)

#%% function version of the script above
import subprocess
import os


def run_simulation(basepath, i_range, j_range, k_range, feb_path_func, vtk_path_func):
    """
    Run a simulation using febio4 based on the specified ranges for i, j, and k.

    Parameters:
    - basepath (str): The base directory where the .feb files are stored.
    - i_range (tuple): A tuple representing the range for i (start, end).
    - j_range (tuple): A tuple representing the range for j (start, end).
    - k_range (tuple): A tuple representing the range for k (start, end).
    - feb_path_func (function): A function that generates feb_path based on i, j, k.
    - vtk_path_func (function): A function that generates vtk_path based on i, j, k.

    Returns:
    - None
    """
    for i in range(*i_range):
        for j in range(*j_range):
            for k in range(*k_range):
                # Generate paths using the provided functions
                feb_path = feb_path_func(basepath, i, j, k)
                vtk_path = vtk_path_func(basepath, i, j, k)

                # Write the command line and run it
                cmd = ["febio4", "run", "-i", feb_path, "-p", vtk_path]
                print("Running:", " ".join(cmd))

                result = subprocess.run(cmd, capture_output=True, text=True)

                # Error handling
                if result.returncode != 0:
                    print(f"Error at index {i}_{j}_{k}:\n{result.stderr}")
                else:
                    print(result.stdout)

# functions to generate paths (change the name of the files accordingly)
def generate_feb_path(basepath, i, j, k):
    return os.path.join(basepath, f"Whole_heart_2016_42_mesh_V3_PreSim_{k}_{j}_{i}.feb")

def generate_vtk_path(basepath, i, j, k):
    return os.path.join(basepath, "jobs", f"Output_WH_{k}_{j}_{i}.vtk")

# Example usage:
basepath = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation"
run_simulation(basepath, (0, 11), (0, 6), (0, 6), generate_feb_path, generate_vtk_path)



#%% Extracting result
import os
#set the folder where you stock the .vtk file as base path
basepath = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/jobs"

for i in range(0,11):
    for j in range(6):
        for k in range(6):
            # Applying the slicer
            path_3D = basepath
            output_dir1 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/jobs/sliced_CINES_29"
            input_seg1 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk"
            file3D_pattern = f"Output_WH_{k}_{j}_{i}.{{t}}.vtk"
            seg_pattern = "rotated_{t}.vtk"
            output_pattern = f"transverse_slice_{k}_{j}_{i}.{{t:02d}}.vtk"
            step = 40
            vtk2Dslicer(path_3D, input_seg1, output_dir1, file3D_pattern, seg_pattern, output_pattern, step)
            input_seg2 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_30/AO_SINUS_STACK_CINES_30_vtk"
            output_dir2 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/jobs/sliced_CINES_30"
            vtk2Dslicer(path_3D, input_seg2, output_dir2, file3D_pattern, seg_pattern, output_pattern, step)
            input_seg3 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_31/AO_SINUS_STACK_CINES_31_vtk"
            output_dir3 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/jobs/sliced_CINES_31"
            vtk2Dslicer(path_3D, input_seg3, output_dir3, file3D_pattern, seg_pattern, output_pattern, step)
            input_seg4 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_32/AO_SINUS_STACK_CINES_32_vtk"
            output_dir4 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/jobs/sliced_CINES_32"
            vtk2Dslicer(path_3D, input_seg4, output_dir4, file3D_pattern, seg_pattern, output_pattern, step)
            input_seg5 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_33/AO_SINUS_STACK_CINES_33_vtk"
            output_dir5 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/jobs/sliced_CINES_33"
            vtk2Dslicer(path_3D, input_seg5, output_dir5, file3D_pattern, seg_pattern, output_pattern, step)
            input_seg6 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_34/AO_SINUS_STACK_CINES_34_vtk"
            output_dir6 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/jobs/sliced_CINES_34"
            vtk2Dslicer(path_3D, input_seg6, output_dir6, file3D_pattern, seg_pattern, output_pattern, step)
            #LVOT segmentations have a different seg pattern because they include 7 key points (surfaces)
            input_seg7 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_24.nii/LVOT_SSFP_CINE_24_vtk"
            output_dir7 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/jobs/sliced_CINES_LVOT_24"
            seg_pattern_LVOT = "rotated_surface_1_time_{t}.vtk"
            vtk2Dslicer(path_3D, input_seg7, output_dir7, file3D_pattern, seg_pattern_LVOT, output_pattern, step)
            input_seg8 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25.nii/LVOT_SSFP_CINE_25_vtk"
            output_dir8 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/jobs/sliced_CINES_LVOT_25"
            vtk2Dslicer(path_3D, input_seg8, output_dir8, file3D_pattern, seg_pattern_LVOT, output_pattern, step)




# %%
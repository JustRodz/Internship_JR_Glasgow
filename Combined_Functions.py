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





#%% Miscellanous function such as : updates_BC_param, updates_output, warp_vtk, calculate_svd_normal, write_ feb and run_febio
# Miscellanous function such as : updates_BC_param, updates_output, warp_vtk, calculate_svd_normal, write_ feb and run_febio

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




#Useful to modify the values of a specified B.C
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


############################################################
####### Test 1 #############################################
############################################################
## Test the node extraction & other pleliminary functions ##

# input_file_path = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_PreSim.feb"
# node_of_interest=find_presc_disp_node(input_file_path)
# nodes_data= extract_nodeset_data(input_file_path, node_of_interest)
# node_coordinates = extract_node_coordinates(input_file_path)
# print("Coordinates :")
# print(node_of_interest)
# print("Finished the search")
# print(nodes_data[0][3])
# print("Finished extracting node`s data")
# original_coord_system = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) #We theorise that the feb file coordinate system is the orthonormal direct system
# BC_coord_system = nodes_data[0][3]
# transform_matrix = calc_rota_matrix(BC_coord_system, original_coord_system)
# print(transform_matrix)

############################################################
####### Test 2 #############################################
############################################################
## Test the update B.C part ##

# updates = {'initial_value': 0, 'relative': 0}
# file_path = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_PreSim_modifier_test.feb"
# update_BC_parameters(file_path, "PrescribedDisplacement2", updates)

#Useful to modify the type of output file
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

# updates = {'new_type': 'vtk'}
# input_dir = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation"
# for i in range (-5,6):
#     for j in range (-5, 6):
#         for k in range (-2, -1):
#             file_path = os.path.join(input_dir, f"Whole_heart_2016_42_mesh_V3_PreSim_{i}_{j}_{k}.feb")
#             update_output(file_path, updates)


def warp_vtk(input_file, output_file):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(input_file)
    reader.Update()
    # Create the warp vector filter and set its input
    warp_vector = vtk.vtkWarpVector()
    warp_vector.SetInputData(reader.GetOutput())

    # Update the warp vector filter to apply the warp
    warp_vector.Update()

    # Write the warped model to a new VTK file
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(warp_vector.GetOutput())
    writer.Write()
    # print(f'Processed : {output_file}')



def write_feb(reference, NewFebName, params = None): 
    for i in range(len(params)):
        update_BC_parameters(reference, params[i][0], params[i][1], NewFebName)
    updates_file_output = {'new_type': 'vtk'}
    update_output(NewFebName, updates_file_output, NewFebName)
    return



def run_febio(file_path, file_name):
    print("Running FEBio simulation", flush=True)

    try:
        # Command to set environment variable
        os.system('export OMP_NUM_THREADS=2')
        feb_file = os.path.join(file_path, file_name+ '.feb')
        log_file = os.path.join(file_path, file_name+ '.log')
        # Command to run FEBio
        result = os.system(f'febio4 {feb_file} -o {log_file}')
        print(f"Command executed with return code: {result}")
    except Exception as e:
        print(f"Could not run FEBio or some other error: {e}", flush=True)
        return
    log_file_path = log_file
    print(f"Attempting to open log file at: {log_file_path}")

    try:
        with open(log_file_path, 'r') as logf:
            lines = logf.readlines()
            # Check if the last non-empty line contains the termination message
            last_line = next((line for line in reversed(lines) if line.strip()), "")
            if "N O R M A L   T E R M I N A T I O N" in last_line:
                print("FEBio simulation completed", flush=True)
                vtk_file = os.path.join(file_path, file_name+ '.0.vtk')
                return vtk_file
            else:
                raise RuntimeError("Error: FEBio simulation did not end normally. Exiting.")
    except Exception as e:
        print(f"Error reading log file: {e}")

#%% Script using Update_BC_parameters to adjust our simulation

input_file_path = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_PreSim.feb"
node_of_interest=find_presc_disp_node(input_file_path)
nodes_data= extract_nodeset_data(input_file_path, node_of_interest)
node_coordinates = extract_node_coordinates(input_file_path)
original_coord_system = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) #We theorise that the feb file coordinate system is the orthonormal direct system
BC_coord_system = nodes_data[0][3]
R = calc_rota_matrix(BC_coord_system, original_coord_system)

file_path = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Whole_heart_2016_42_mesh_V3_PreSim_modifier_test.feb"
output_dir= "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation" 
for i in range (-5,6):
    for j in range (-5, 6):
        for k in range (-2, -1):
            coords =np.array([i, j, k])
            coords_xyz = np.dot(R, coords) 
            output = os.path.join(output_dir, f"Whole_heart_2016_42_mesh_V3_PreSim_{i}_{j}_{k}.feb")
            x_bc = {'initial_value': coords_xyz[0]}
            update_BC_parameters(file_path, "PrescribedDisplacement2", x_bc, output)
            y_bc = {'initial_value': coords_xyz[1]}
            update_BC_parameters(output, "PrescribedDisplacement3", y_bc)
            z_bc = {'initial_value': coords_xyz[2]}
            update_BC_parameters(output, "PrescribedDisplacement4", z_bc)




#%% Saving the header of the segmentation (useful to rotate the model)
def Seg_header(file_path, seg_path, output_path):
    affine = nib.load(file_path).affine  ##retrieves the affine transformation matrix from the NIfTI file (image coord -> IRL coord)
    mask = nib.load(seg_path).get_fdata()  ##Get_fdata transforms NifTy image data into a NumPy array
    ##plt.imshow(mask[:,:,0,0])
    np.save(output_path,affine)

#%% LvotSeg2vtk
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


#%% Interpolation
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

#%% Seg2contours
# Other Strategy to extract the contour of a segmentation using the Gradient methodology. Main issue = Point are ordered based on their z coordinate (noncontinuous curve) + rougher shape
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




#%% rotate_vtk + rotate_LVOT
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



#%% # %% Generic functions used for extracting data
# #Shape_center + Calculate_distances + Extract_and_fit_curves + vtk2Dslicer

import os
import vtk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev



def vtk2Dslicer (model_path, input_seg_path, output_path, model3D_name_pattern, seg_name_pattern, output_pattern, timesteps):
    """
    model_path (str): Input file for the existing 3D model.
    input_seg_path : Input file for the existing 2d segmentation.
    output_path (str): Output folder for generated .vtk files.
    model3D_name_pattern (str): Pattern for the 3D model filenames, using {i} for the timestep.
    seg_name_pattern (str): Pattern for the rotated segmentation filenames, using {i} for the timestep.
    output_pattern (str): Pattern for the output .vtk filenames
    timesteps (int): Number of timesteps in the model
    """
    os.makedirs(output_path, exist_ok=True)


    for t in range(timesteps):   ##range value is equal to the number of time step, 40 in our case
        vtk_path_model = os.path.join(model_path, model3D_name_pattern.format(t=t))
        # print(f"Reading model: {vtk_path_model}")  # Debug print
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
    # print("vtk2Dslicer - execution completed")



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
    Calculate the Euclidean distances coords by coords between corresponding centers in tab_2DMRI and tab_3Dslice.
    """
    distances = []
    for centre_2D, centre_3D in zip(tab_2DMRI, tab_3Dslice):
        distance = np.linalg.norm(np.array(centre_2D) - np.array(centre_3D))
        distances.append(distance)
    return distances

def calculate_distances_abs(centre_2D, centre_3D):
    """
    Calculate the distances between corresponding centers in tab_2DMRI and tab_3Dslice.
    """
    distance = np.linalg.norm(np.array(centre_2D) - np.array(centre_3D))
    return distance

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


# # Example
# input_path_2DMRI="C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/"
# input_path_3Dslice="C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/"
# tab_2DMRI = np.empty((0, 3), int)
# tab_3Dslice = np.empty((0, 3), int)
# tab_gap= np.empty((0, 3), int)
# for i in range(40) :
#     # Calculate &stock the coordinates of the center of the 2D MRI shape 
#     vtk_file_2DMRI = os.path.join(input_path_2DMRI, f"rotated_{i}.vtk")
#     centre_2DMRI = shape_center(vtk_file_2DMRI)
#     tab_2DMRI = np.vstack([tab_2DMRI, centre_2DMRI])
#     # Calculate &stock the coordinates of the center of the 3D model slice
#     vtk_file_3Dslice = os.path.join(input_path_3Dslice, f"transverse_slice_{i:03d}.vtk")
#     centre_3Dslice = shape_center(vtk_file_3Dslice)
#     tab_3Dslice = np.vstack([tab_3Dslice, centre_3Dslice])
#     # print("At step ", i, " Coordinates of the aorta centre :", centre)
#     points_2DMRI=read_points_vtk(vtk_file_2DMRI)
#     points_3Dslice=read_points_vtk(vtk_file_3Dslice)
#     # gap_2DMRI_3D = gap_calculator_vtk(points_2DMRI, points_3Dslice)
#     # print(f"Gap value at Timestep {i}: {gap_2DMRI_3D}")
#     tab_gap= np.vstack([tab_gap, centre_3Dslice])



# # Calculate distances between centers for each timestep
# distances = calculate_distances(tab_2DMRI, tab_3Dslice)

# # Print the distances
# for i, distance in enumerate(distances):
#     print(f"Distance at timestep {i}: {distance}")


# Extract the extremities in the lvot

# Reordering points if we have a non continuous outline
def reorder_points(points,  distance_threshold):
    """
    This function is required because the slice generated link points by their z-value, thus creating a non-continuous path.

    The funtion reorders a list of 3D points to form a continuous path.
    This method follows a nearest neighbour algorithm:
    it starts from the first point and at each step adds the nearest point
    of those not yet used.
    
    Args:
        points (np.ndarray): array of shapes (N, 3), representing 3D points.
        distance threshold (float): Max distance between two consecutive points
    """

    # # Copy to avoid modifying the original data
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
        # Check if the distance to the next point is within the threshold
        if dists[next_index] < distance_threshold:
            ordered.append(points[next_index])
            used[next_index] = True
        else:
            # If the distance exceeds the threshold, consider it as a separate segment
            break 

    return np.array(ordered)



def remove_consecutive_duplicates(points):
    # Remove consecutive duplicate points based on either x or y coordinate
    mask = np.ones(len(points), dtype=bool)
    for i in range(1, len(points)):
        if points[i, 0] == points[i-1, 0] or points[i, 1] == points[i-1, 1]:
            mask[i] = False
    return points[mask]

def extract_and_fit_curves(filename):
    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    # Get the polydata
    polydata = reader.GetOutput()

    # Extract the coordinates of all points
    points = polydata.GetPoints()
    all_points = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

    # Segment points into separate curves based on distance
    segments = []
    remaining_points = all_points.copy()
    while len(remaining_points) > 0:
        segment = reorder_points(remaining_points, 10.0)
        segments.append(segment)
        remaining_points = np.array([p for p in remaining_points if not np.any(np.all(p == segment, axis=1))])

    # Collect endpoints
    endpoints = []
    for segment in segments:
        endpoints.append((segment[0], segment[-1]))

    # Fit curves to each segment and plot them only in case of error
    for segment in segments:
        segment = remove_consecutive_duplicates(segment)
        x = segment[:, 0]
        y = segment[:, 1]

        if len(x) > 3:  # Ensure there are enough points to fit a spline
            try:
                spline, _ = splprep([x, y], s=len(x)/1000, k=3) # Modify the value of s to adapt the smoothness
                x_fit, y_fit = splev(np.linspace(0, 1, 100), spline)
            except ValueError as e:
                print(f"Error fitting spline for segment: {e}")
                print(f"Segment x values: {x}")
                print(f"Segment y values: {y}")

                # Plot problematic segment
                plt.figure(figsize=(10, 6))
                plt.plot(x, y, 'o-', label=f'Problematic Segment (Length: {len(x)})')
                plt.title('Visualization of Problematic Segments')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.legend()
                plt.grid(True)
                plt.show()

    return endpoints



# # Example usage
# filename = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V3/jobs/sliced_CINES_LVOT_25/transverse_slice_0_1_-2.17.vtk"
# enpoints_test = extract_and_fit_curves(filename)
# print(enpoints_test)


#The following function allow to overwrite (mode = 'w') the specified excel file or to add data to it (mode = 'a')
def save_to_excel(df, excel_output, mode='w', key_columns=None):
    if mode == 'w':
        df.to_excel(excel_output, index=False)
    else:
        try:
            existing_df = pd.read_excel(excel_output)
            combined_df = pd.merge(existing_df, df, on=key_columns, how='outer')
            combined_df.to_excel(excel_output, index=False)
        except FileNotFoundError:
            df.to_excel(excel_output, index=False)

##########################################################################################################################################################################
#%% Code a utiliser avec le cluster
import subprocess
import multiprocessing
import xml.etree.ElementTree as ET
import numpy as np
import vtk
import os
import shutil 
import time
 

def run_simulation(j, i, c, k1, original_febio_file):
    print(f"Running simulation for c={c[j]}, k1={k1[i]}")
    output_febio_file = f"param_sweeping/temp_feb_files/arch_split_axes_{j+1:04d}_{i+1:04d}.feb"
    new_params = {'c': c[j], 'k1': k1[i]}
    # Update material parameters
    update_goh_material_parameters(original_febio_file, 
                                   "Material2", new_params, output_febio_file)
    # Limit FEBio to 1 CPU thread
    os.environ["OMP_NUM_THREADS"] = "1"  # Set the number of CPU threads for each simulation
    # Define the command
    command = f'FEBioStudio/bin/febio4 {output_febio_file} -silent -p param_sweeping/Run_{j+1:04d}_{i+1:04d}/out{i+1:04d}.vtk'
    print("Running FEBio simulation",flush=True)
    try:
        os.system(command)
    except Exception as e:
        print(f"Could not run FEBio: {e}",flush=True)
 
if __name__ == "__main__":
    c = [a/100 for a in range(1, 11, 1)]  # Define your list of c values
    k1 = [a/10 for a in range(1, 11)]  # Define your list of k1 values
    original_febio_file = "param_sweeping/arch_split_axes.feb"
    num_processes = 12
    # num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)
 
    tasks = [(j, i, c, k1, original_febio_file) for j in range(len(c)) for i in range(len(k1))]
 
    # Run simulations in parallel
    pool.starmap(run_simulation, tasks)
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()


#%% Simulation runner
import subprocess
import os

##set the folder where you stock the .feb file as base path
basepath = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation"

#adjust the ranges to coorespond with the simu
for x in range(1):
    for y in range(1):
        for z in range(-10,11):
            #Specify input and output
            feb_path = os.path.join(basepath, f"Whole_heart_2016_42_mesh_V3_PreSim_{x}_{y}_{z}.feb")
            vtk_path = os.path.join(basepath,"Simulation_V2", "jobs", f"Output_WH_{x}_{y}_{z}.vtk")

            # write the command line and run it
            cmd = ["febio4", "run", "-i", feb_path, "-p", vtk_path]
            print("Running:", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)

            #error case
            if result.returncode != 0:
                print(f"Error at index {z}:\n{result.stderr}")
            else:
                print(result.stdout)

#%% function version of the script above
import subprocess
import os


def run_simulation(basepath, x_range, y_range, z_range, feb_path_func, vtk_path_func):
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
    for x in range(*x_range):
        for y in range(*y_range):
            for z in range(*z_range):
                # Generate paths using the provided functions
                feb_path = feb_path_func(basepath, x, y, z)
                vtk_path = vtk_path_func(basepath, x, y, z)

                # Write the command line and run it
                cmd = ["febio4", "run", "-i", feb_path, "-p", vtk_path]
                print("Running:", " ".join(cmd))

                result = subprocess.run(cmd, capture_output=True, text=True)

                # Error handling
                if result.returncode != 0:
                    print(f"Error at index {x}_{y}_{z}:\n{result.stderr}")
                else:
                    print(result.stdout)

# functions to generate paths (change the name of the files accordingly)
def generate_feb_path(basepath, x, y, z):
    return os.path.join(basepath, f"Whole_heart_2016_42_mesh_V3_PreSim_{x}_{y}_{z}.feb")

def generate_vtk_path(basepath, x, y, z):
    return os.path.join(basepath,"Simulation_V3", "jobs", f"Output_WH_{x}_{y}_{z}.vtk")

# Example usage:
basepath = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation"
run_simulation(basepath, (-2, -1), (-5, 6), (-5, 6), generate_feb_path, generate_vtk_path)



#%% To extract data we need to apply the "warp by vector" function to our file
def warp_vtk_range(x_range, y_range, z_range, timesteps, input_pattern, output_pattern):
    for x in x_range:
        for y in y_range:
            for z in z_range:
                for t in timesteps:  
                    reader = vtk.vtkUnstructuredGridReader()
                    input_file = input_pattern.format(x=x, y=y, z=z, t=t)
                    output_file = output_pattern.format(x=x, y=y, z=z, t=t)
                    reader.SetFileName(input_file)
                    reader.Update()
                    # Create the warp vector filter and set its input
                    warp_vector = vtk.vtkWarpVector()
                    warp_vector.SetInputData(reader.GetOutput())
        
                    # Update the warp vector filter to apply the warp
                    warp_vector.Update()
        
                    # Write the warped model to a new VTK file
                    writer = vtk.vtkUnstructuredGridWriter()
                    writer.SetFileName(output_file)
                    writer.SetInputData(warp_vector.GetOutput())
                    writer.Write()
                
        
                    print(f'Processed : Output_{x}_{y}_{z}.{t}.vtk')

#Arguments
x_values = range(0, 1)
y_values = range(0, 1)
z_values = range(-10, 11)
timestep_values = range(40)
input_pattern = f"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/Output_WH_{x}_{y}_{z}.{t}.vtk"
output_pattern = f"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/Output_WH_Warped_{x}_{y}_{z}.{t}.vtk"
warp_vtk_range(x_values, y_values, z_values, timestep_values, input_pattern, output_pattern)



#%% Extracting slice : script
import os
#set the folder where you stock the .vtk file as base path
basepath = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs"

for i in range(-2, -1):
    for j in range(-5, 6):
        for k in range(-5, 6):
            # Applying the slicer
            path_3D = basepath
            output_dir1 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_29"
            input_seg1 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk"
            file3D_pattern = f"Output_WH_Warped_{k}_{j}_{i}.{{t}}.vtk"
            seg_pattern = "rotated_{t}.vtk"
            output_pattern = f"transverse_slice_{k}_{j}_{i}.{{t:02d}}.vtk"
            step = 40
            vtk2Dslicer(path_3D, input_seg1, output_dir1, file3D_pattern, seg_pattern, output_pattern, step)
            input_seg2 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_30/AO_SINUS_STACK_CINES_30_vtk"
            output_dir2 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_30"
            vtk2Dslicer(path_3D, input_seg2, output_dir2, file3D_pattern, seg_pattern, output_pattern, step)
            input_seg3 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_31/AO_SINUS_STACK_CINES_31_vtk"
            output_dir3 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_31"
            vtk2Dslicer(path_3D, input_seg3, output_dir3, file3D_pattern, seg_pattern, output_pattern, step)
            input_seg4 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_32/AO_SINUS_STACK_CINES_32_vtk"
            output_dir4 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_32"
            vtk2Dslicer(path_3D, input_seg4, output_dir4, file3D_pattern, seg_pattern, output_pattern, step)
            input_seg5 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_33/AO_SINUS_STACK_CINES_33_vtk"
            output_dir5 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_33"
            vtk2Dslicer(path_3D, input_seg5, output_dir5, file3D_pattern, seg_pattern, output_pattern, step)
            input_seg6 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_34/AO_SINUS_STACK_CINES_34_vtk"
            output_dir6 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_34"
            vtk2Dslicer(path_3D, input_seg6, output_dir6, file3D_pattern, seg_pattern, output_pattern, step)
            #LVOT segmentations have a different seg pattern because they include 7 key points (surfaces)
            input_seg7 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_24/LVOT_SSFP_CINE_24_vtk"
            output_dir7 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_LVOT_24"
            seg_pattern_LVOT = "rotated_surface_2_time_{t:02d}.vtk"  ### Don`t use "rotated_surface_1_time_{t:02d}.vtk" as it doesn`t exist for the LVOT_SSFP_CINES_24 segmentation (no junctoin with the aorta arch in this plane)
            vtk2Dslicer(path_3D, input_seg7, output_dir7, file3D_pattern, seg_pattern_LVOT, output_pattern, step)
            input_seg8 = r"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25/LVOT_SSFP_CINE_25_vtk"
            output_dir8 = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_LVOT_25"
            vtk2Dslicer(path_3D, input_seg8, output_dir8, file3D_pattern, seg_pattern_LVOT, output_pattern, step)


#%% Extracting slice : Function version
###################### Function version
import os
def extract_slice(basepath, x_range, y_range, z_range, input_3D_pattern, input_2DMRI_pattern, output_vtk_pattern, timestep, output_dir_path, input_seg_path):
    for x in range(*x_range):
        for y in range(*y_range):
            for z in range(*z_range):
                path_3D = basepath
                file3D_pattern = input_3D_pattern.format(x=x, y=y, z=z)
                # seg_pattern = input_2DMRI_pattern.format(x=x, y=y, z=z, t=t)
                output_pattern = output_vtk_pattern.format(x=x, y=y, z=z)
                step = timestep
                output_dir = output_dir_path
                input_seg = input_seg_path
                vtk2Dslicer(path_3D, input_seg, output_dir, file3D_pattern, seg_pattern, output_pattern, step)
                print(f'Processed : Output_{x}_{y}_{z}.vtk')

# Horizontal segmentation
# basepath = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs"
# x_range = (0, 1)
# y_range = (0, 1)
# z_range = (-10, 11)
# file3D_pattern = r"Output_WH_Warped_{x}_{y}_{z}.{{t}}.vtk"
# seg_pattern = r"rotated_{t}.vtk"
# output_pattern = r"transverse_slice_{x}_{y}_{z}.{{t:02d}}.vtk"
# step = 40
# for cine in [29,30, 31, 32, 33, 34]:
#     output_dir = f"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_{cine}"
#     input_seg = f"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_{cine}/AO_SINUS_STACK_CINES_{cine}_vtk"
#     extract_slice(basepath, x_range, y_range, z_range, file3D_pattern, seg_pattern, output_pattern, step, output_dir, input_seg)

# LVOT Segmentation
basepath = r"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs"
x_range = (0, 1)
y_range = (0, 1)
z_range = (-10, 11)
file3D_pattern = r"Output_WH_Warped_{x}_{y}_{z}.{{t}}.vtk"
seg_pattern = r"rotated_surface_2_time_{t:02d}.vtk"     #Use surface_2 because surface_1 is void for LVOT_SSFP_CINES_24
output_pattern = r"transverse_slice_{x}_{y}_{z}.{{t:02d}}.vtk"
step = 40
for cine in [24, 25]:
    output_dir = f"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_LVOT_{cine}"
    input_seg = f"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_{cine}/LVOT_SSFP_CINE_{cine}_vtk"
    extract_slice(basepath, x_range, y_range, z_range, file3D_pattern, seg_pattern, output_pattern, step, output_dir, input_seg)

#############################################################################################################################################################################



#%% Calculate distance & save them in Excel
import os
import numpy as np
import pandas as pd

# Execute shape_center, read_points_vtk, and calculate_distances before

def distance2excel_range(xrange, yrange, zrange, timesteps, cines, basepath_2dMRI, path_2d_pattern, file_2d_pattern, basepath_3dmodel, path_3d_pattern, excel_output=None, mode='w', key_columns=['x', 'y', 'z', 't']):
    # Initialize a dictionary to store all data
    all_data = []
    
    for x in xrange:
        for y in yrange:
            for z in zrange:
                for t in timesteps:
                    row_data = {'x': x, 'y': y, 'z': z, 't': t}

                    for cine in cines:
                        input_path_2DMRI = os.path.join(basepath_2dMRI, path_2d_pattern.format(cine = cine))
                        input_path_3Dslice = os.path.join(basepath_3dmodel, path_3d_pattern.format(cine = cine))

                        # Extracting center coords from the 2D MRI
                        vtk_file_2DMRI = os.path.join(input_path_2DMRI, file_2d_pattern.format(t=t) )
                        centre_2DMRI = shape_center(vtk_file_2DMRI)

                        # Extracting center coords from the slice from the 3D model
                        vtk_file_3Dslice = os.path.join(input_path_3Dslice, file_3d_pattern.format(x=x, y=y, z=z, t=t) )
                        centre_3Dslice = shape_center(vtk_file_3Dslice)

                        distance_coords = calculate_distances(centre_2DMRI, centre_3Dslice)
                        distance_abs = calculate_distances_abs(centre_2DMRI, centre_3Dslice)
                        
                        # row_data[f'CINES_{cine}_centre_2dMRI'] = centre_2DMRI
                        # row_data[f'CINES_{cine}_centre_3d_model'] = centre_3Dslice
                        row_data[f'distance_xyz_CINES_{cine}'] = distance_coords
                        row_data[f'distance_abs_CINES_{cine}'] = distance_abs


                    all_data.append(row_data)
    
    # Create a DataFrame from the list of data
    df = pd.DataFrame(all_data)
    if excel_output is not None:
        # Save the DataFrame to an Excel file
        save_to_excel(df, excel_output, mode, key_columns)

    return(df)

def distance2excel(timesteps, cines, basepath_2dMRI, path_2d_pattern, file_2d_pattern, basepath_3dmodel, path_3d_pattern, file_3d_pattern, excel_output=None, mode='w', key_columns=['x', 'y', 'z', 't']):
    # Initialize a dictionary to store all data
    all_data = []
    
    for t in range(timesteps):
        row_data = {'t': t}

        for cine in cines:
            input_path_2DMRI = os.path.join(basepath_2dMRI, path_2d_pattern.format(cine = cine))
            input_path_3Dslice = os.path.join(basepath_3dmodel, path_3d_pattern.format(cine = cine))

            # Extracting center coords from the 2D MRI
            vtk_file_2DMRI = os.path.join(input_path_2DMRI, file_2d_pattern.format(t=t) )
            # print("vtk_file_2DMRI : ", vtk_file_2DMRI)
            centre_2DMRI = shape_center(vtk_file_2DMRI)

            # Extracting center coords from the slice from the 3D model
            vtk_file_3Dslice = os.path.join(input_path_3Dslice, file_3d_pattern.format(t=t) )
            # print("vtk_file_3Dslice : ", vtk_file_3Dslice)
            centre_3Dslice = shape_center(vtk_file_3Dslice)

            distance_coords = calculate_distances(centre_2DMRI, centre_3Dslice)
            distance_abs = calculate_distances_abs(centre_2DMRI, centre_3Dslice)
            
            # row_data[f'CINES_{cine}_centre_2dMRI'] = centre_2DMRI
            # row_data[f'CINES_{cine}_centre_3d_model'] = centre_3Dslice
            row_data[f'distance_xyz_CINES_{cine}'] = distance_coords
            row_data[f'distance_abs_CINES_{cine}'] = distance_abs


        all_data.append(row_data)
    
    # Create a DataFrame from the list of data
    df = pd.DataFrame(all_data)
    if excel_output is not None:
        # Save the DataFrame to an Excel file
        save_to_excel(df, excel_output, mode, key_columns)

    return(df)
# #test
# xrange = range(0, 1)
# yrange = range(0, 1)
# zrange = range(-10, 11)
# timesteps = range(0, 40)
# cines = [29, 30, 31, 32, 33, 34]
# basepath_2dMRI = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D"
# path_2d_pattern = "AO_SINUS_STACK_CINES_{cine}/AO_SINUS_STACK_CINES_{cine}_vtk"
# file_2d_pattern = "rotated_{t}.vtk"
# basepath_3dmodel = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs"
# path_3d_pattern = "sliced_CINES_{cine}"
# file_3d_pattern = "transverse_slice_{x}_{y}_{z}.{t:02d}.vtk"
# excel_output = "center_distance_data_test.xlsx"
# mode = 'a'
# distance2excel_range(xrange, yrange, zrange, timesteps, cines, basepath_2dMRI, path_2d_pattern, file_2d_pattern, basepath_3dmodel, path_3d_pattern, excel_output)
###################################################################################################################

#%% Calculate distance at diastole and systole modified version of the script above
import os
import numpy as np
import pandas as pd

# Execute shape_center, read_points_vtk, and calculate_distances

# Initialize a dictionary to store all data
all_data = []

for x in range(0, 1):
    for y in range(0, 1):
        for z in range(-10, 11):
            for t in [0, 1, 17]:
                row_data = {'x': x, 'y': y, 'z': z, 't': t}
                t_model = 39 if t == 17 else t
                for cine in [29, 30, 31, 32, 33, 34]:
                    input_path_2DMRI = f"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_{cine}/AO_SINUS_STACK_CINES_{cine}_vtk"
                    input_path_3Dslice = f"C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_{cine}"
                    #Extracting center coords from the 2D MRI
                    vtk_file_2DMRI = os.path.join(input_path_2DMRI, f"rotated_{t}.vtk")
                    centre_2DMRI = shape_center(vtk_file_2DMRI)
                    # print(vtk_file_2DMRI)
                    #Extracting center coords from the slice from the 3D model
                    vtk_file_3Dslice = os.path.join(input_path_3Dslice, f"transverse_slice_{x}_{y}_{z}.{t_model:02d}.vtk")
                    print(vtk_file_3Dslice)
                    centre_3Dslice = shape_center(vtk_file_3Dslice)


                    points_2DMRI = read_points_vtk(vtk_file_2DMRI)
                    points_3Dslice = read_points_vtk(vtk_file_3Dslice)
                
                    distance = calculate_distances_abs(centre_2DMRI, centre_3Dslice)
                    print(distance)

                    row_data[f'distance_CINES_{cine}'] = distance
                    
                all_data.append(row_data)


# Create a DataFrame from the list of data
df = pd.DataFrame(all_data)

# Save the DataFrame to an Excel file
df.to_excel("center_distance_data_Z_values_systole_diastole.xlsx", index=False)
#####################################################################################################################################################

#%% Calculte the norm of data stocked in an excel file and return a new excel
import pandas as pd
def calcul_norm_excel(excel_path, columns_name, output_excel=None):
    # load the excel file
    file_path = excel_path
    data = pd.read_excel(file_path)
    # Specify the columns used
    distance_columns = columns_name
    # Calculate the norm for each line
    data['distance_norm'] = data[distance_columns].apply(lambda row: (row**2).sum()**0.5, axis=1)
    # Save data in a new excel file
    output_file_path = output_excel if output_excel else file_path
    data.to_excel(output_file_path, index=False)
    print(f"Results have been saved in {output_file_path}")


file_input_path = "center_distance_data_Z_values_systole_diastole.xlsx"  # Replace as needed
# output_file_path ="center_distance_data_XY_values_Out_testing.xlsx"  # Replace as needed
distance_columns = ['distance_CINES_30', 'distance_CINES_31', 'distance_CINES_32', 'distance_CINES_33', 'distance_CINES_34']

calcul_norm_excel(file_input_path, distance_columns)
############################################################################################################################################################
#%% plot distance_norm/z (z=i in the dataset)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the Excel file
file_path = "C:/Users/jr403s/Documents/code/center_distance_data_Z_values_systole_diastole.xlsx"
data = pd.read_excel(file_path)

for x in range(1):
    for y in range(1):
        # Filter data for t=1 (diastole) or t = 17 (systole)
        t_constant_data = data[(data['t'] == 17) & (data['x'] == x) & (data['y'] == y)].copy()

        # Plot the relationship between k and distance_norm
        fig1 = px.scatter(t_constant_data, x='z', y='distance_norm_V2',
                        title=f"Relation between z and the norm of the distances at t=17 (systole)",
                        labels={'z': 'z values', 'distance_norm_V2': 'Norm of the distances'})  

        # Show the plot
        print("For x = ", x, " and y = ", y )
        fig1.show()

#################################################################################################################

#%% function for least square fitting and finding the minimum
import pandas as pd
import numpy as np
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt

def fit_model_and_find_minimum(t, file_path, first_col, second_col, model_func, initial_params, fixed_col=None):
    """
    Fit a model to data and find the minimum of the fitted model.

    Parameters:
    - t: The fixed value of t to filter the data.
    - file_path: Path to the Excel file.
    - first_col: Name of the column containing first values (z here) which serves for the X-axis.
    - second_col: Name of the column containing the second values (distance here) which serves for the Y-axis .
    - fixed_col: Name of the column containing the fixed value (t here).
    - model_func: The model function to fit.
    - initial_params: Initial parameters for the model.

    Returns:
    - A tuple containing optimal parameters, minimum value information, and the plot.
    """
    # Load data from the Excel file
    data = pd.read_excel(file_path)

    # Filter data for the fixed value of t if fixed_col is provided
    if fixed_col is not None:
        fixed_data = data[data[fixed_col] == t].copy()
    else:
        fixed_data = data.copy()

    # Extract columns z and distance
    z = fixed_data[first_col].values
    distance = fixed_data[second_col].values

    # Residual function for the model
    def residuals(params, z, distance):
        return distance - model_func(z, *params)

    # Least squares optimization
    result = least_squares(residuals, initial_params, args=(z, distance))

    # Optimal parameters
    optimal_params = result.x
    print(f"For t = {t}")
    print(f"Optimal parameters: {optimal_params}")

    # Find the minimum of the fitted model
    result_min = minimize(lambda z: model_func(z, *optimal_params), x0=0)

    # Minimum of the model
    z_min = result_min.x[0]
    distance_min = result_min.fun
    print(f"Minimum of the model at z = {z_min}: distance = {distance_min}")

    # Plot the data and the fitted model
    plt.scatter(z, distance, label='Data')
    z_fit = np.linspace(min(z), max(z), 100)
    distance_fit = model_func(z_fit, *optimal_params)
    plt.plot(z_fit, distance_fit, label='Fitted model', color='red')
    plt.scatter(z_min, distance_min, color='green', label=f'Minimum: ({z_min:.2f}, {distance_min:.2f})')
    plt.xlabel(first_col)
    plt.ylabel(second_col)
    plt.title(f'Model fitted for t = {t}')
    plt.legend()
    plt.show()

    return optimal_params, (z_min, distance_min), plt


# test
t = 1
file_path = "C:/Users/jr403s/Documents/code/Simulation_complete_data_test.xlsx"
first_col = 't'
second_col = 'Strech_3d'
fixed_col = 'z'
model_func = lambda z, a, b, c: a * (z**2) + b * z + c
initial_params = [1.0, 1.0, 1.0]

optimal_params, min_info, plot = fit_model_and_find_minimum(t, file_path, first_col, second_col, model_func, initial_params, fixed_col)

####################################################################################################################################################

#%% Test, max strectch depending on t
from scipy.optimize import least_squares, minimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fit_model_and_find_maximum(t, file_path, first_col, second_col, model_func, initial_params, fixed_col=None):
    """
    Fit a model to data and find the maximum of the fitted model.
    Parameters:
    - t: The fixed value of t to filter the data.
    - file_path: Path to the Excel file.
    - first_col: Name of the column containing first values (z here) which serves for the X-axis.
    - second_col: Name of the column containing the second values (distance here) which serves for the Y-axis.
    - fixed_col: Name of the column containing the fixed value (t here).
    - model_func: The model function to fit.
    - initial_params: Initial parameters for the model.
    Returns:
    - A tuple containing optimal parameters, maximum value information, and the plot.
    """
    # Load data from the Excel file
    data = pd.read_excel(file_path)

    # Filter data for the fixed value of t if fixed_col is provided
    if fixed_col is not None:
        fixed_data = data[data[fixed_col] == t].copy()
    else:
        fixed_data = data.copy()

    # Extract columns z and distance
    z = fixed_data[first_col].values
    distance = fixed_data[second_col].values

    # Residual function for the model
    def residuals(params, z, distance):
        return distance - model_func(z, *params)

    # Least squares optimization
    result = least_squares(residuals, initial_params, args=(z, distance))

    # Optimal parameters
    optimal_params = result.x
    print(f"For z = {t}")
    print(f"Optimal parameters: {optimal_params}")

    # Find the maximum of the fitted model by minimizing the negative of the model
    result_max = minimize(lambda z: -model_func(z, *optimal_params), x0=0)

    # Maximum of the model
    z_max = result_max.x[0]
    distance_max = -result_max.fun
    print(f"Maximum of the model at t = {z_max}: Stretch = {distance_max}")

    # Plot the data and the fitted model
    plt.scatter(z, distance, label='Data')
    z_fit = np.linspace(min(z), max(z), 100)
    distance_fit = model_func(z_fit, *optimal_params)
    plt.plot(z_fit, distance_fit, label='Fitted model', color='red')
    plt.scatter(z_max, distance_max, color='green', label=f'Maximum: ({z_max:.2f}, {distance_max:.2f})')
    plt.xlabel(first_col)
    plt.ylabel(second_col)
    plt.title(f'Model fitted for z = {t}')
    plt.legend()
    plt.show()

    return optimal_params, (z_max, distance_max), plt

# Test
t = 0
file_path = "C:/Users/jr403s/Documents/code/Simulation_complete_data_test.xlsx"
first_col = 't'
second_col = 'Strech_2d'
fixed_col = 'z'
model_func = lambda t, a, b, c, d, e, f, g: a * (t**6) + b * (t**5) + c * (t**4) + d * (t**3) + e * (t**2) + f * t + g
initial_params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
optimal_params, max_info, plot = fit_model_and_find_maximum(t, file_path, first_col, second_col, model_func, initial_params, fixed_col)

#%% function plotting the contour graph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_contour_and_find_min(file_path, t_value, X_col, Y_col, distance_col):
    """
    Plot a contour graph and find the minimum distance value.

    Parameters:
    - file_path: Path to the Excel file.
    - t_value: The fixed value of t to filter the data.
    - X_col: Name of the column containing X values.
    - Y_col: Name of the column containing Y values.
    - distance_col: Name of the column containing distance values.
    """
    # Load data from the Excel file
    data = pd.read_excel(file_path)

    # Filter data for t = t_value
    fixed_data = data[data['t'] == t_value].copy()

    # Extract necessary columns
    x_values = fixed_data[X_col]
    y_values = fixed_data[Y_col]
    distance_values = fixed_data[distance_col]

    # Create a grid for x and y
    x_grid = np.linspace(min(x_values), max(x_values), 100)
    y_grid = np.linspace(min(y_values), max(y_values), 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Interpolate distance values onto the grid
    distance_grid = griddata((x_values, y_values), distance_values, (X, Y), method='cubic')

    # Find the minimum interpolated distance value
    min_distance = np.min(distance_grid)
    min_index = np.unravel_index(np.argmin(distance_grid), distance_grid.shape)
    min_x, min_y = X[min_index], Y[min_index]

    # Create the contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, distance_grid, levels=20, cmap='copper')
    plt.colorbar(contour, label=distance_col)

    # Add labels and a title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Contour plot of {distance_col} = f(x, y) at t={t_value}')

    # Display the point of minimum distance
    plt.scatter(min_x, min_y, color='red', label=f'Minimum: ({min_x:.2f}, {min_y:.2f}) = {min_distance:.2f}')
    plt.legend()

    # Show the plot
    plt.show()

    # Return the minimum distance value and its coordinates
    return min_distance, (min_x, min_y)

# Example usage of the function
file_path = 'center_distance_data_XY_values_testing.xlsx'
t_value = 1
distance_col = 'distance_norm'
X_col = 'k'
Y_col = 'j'
min_distance, min_coords = plot_contour_and_find_min(file_path, t_value, X_col, Y_col, distance_col)
print(f"Minimum distance value: {min_distance:.3f} at coordinates (x, y) = ({min_coords[0]:.3f}, {min_coords[1]:.3f})")

####################################################################################################################################################################




#%% test calcul height

import os
import pandas as pd

def LVOT_2d_height2excel(xrange, yrange, zrange, timesteps, basepath, top_point_pattern, bottom_point_pattern, excel_output, mode='w', key_columns=['x', 'y', 'z', 't']):
    all_data = []
    for x in xrange:
        for y in yrange:
            for z in zrange:
                for t in (timesteps):
                    row_data = {'x': x, 'y': y, 'z': z, 't': t}
                    vtk_file_top_point = os.path.join(basepath, top_point_pattern.format(t=t) )
                    top_point = shape_center(vtk_file_top_point)

                    vtk_file_bottom_point = os.path.join(basepath, bottom_point_pattern.format(t=t))
                    bottom_point = shape_center(vtk_file_bottom_point)
                    
                    height_t = calculate_distances_abs(top_point, bottom_point)
                    if t== timesteps[0]:
                        h_init = height_t
                    row_data['height_2d_model'] = height_t
                    row_data['Strech_2d'] = (height_t - h_init)/ h_init
                    all_data.append(row_data)
                    # print("for x = ", x," for y = ", y," for z = ", z, " for t = ", t, ", the height = ", height_t)

    # Create a DataFrame with the height data
    df = pd.DataFrame(all_data)

    # Save the DataFrame to an Excel file
    save_to_excel(df, excel_output, mode, key_columns)

# # Example usage:
# xrange = range(0, 1)
# yrange = range(0, 1)
# zrange = range(-10, 11)
# timesteps = range(0, 40)
# basepath = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25/LVOT_SSFP_CINE_25_vtk"
# top_point = r"rotated_surface_1_time_{t:02d}.vtk"
# bottom_point = r"rotated_surface_2_time_{t:02d}.vtk"
# excel_output = "height_2Ddata_stretch.xlsx"
# mode = 'a'
# key_columns=['x', 'y', 'z', 't']
# LVOT_2d_height2excel(xrange, yrange, zrange, timesteps, basepath, top_point, bottom_point, excel_output, mode)

####################################################################################################################################################################
#%% LVOT height 3D model
import os
import pandas as pd
import numpy as np

def LVOT_3D_height2excel(xrange, yrange, zrange, timesteps, basepath, vtk_file_pattern, excel_output, mode='w', key_columns=['x', 'y', 'z', 't']):
    all_data = []
    for x in xrange:
        for y in yrange:
            for z in zrange:
                for t in timesteps:
                    row_data = {'x': x, 'y': y, 'z': z, 't': t}
                    output_pattern = vtk_file_pattern.format(x=x, y=y, z=z, t=t)
                    filename = os.path.join(basepath, output_pattern)
                    endpoints_test = extract_and_fit_curves(filename)
                    height_loc3d = []
                    for l in range(len(endpoints_test)):
                        height_loc3d.append(calculate_distances_abs(endpoints_test[l][0],endpoints_test[l][1]))
                    height_moy=np.mean(height_loc3d)
                    if t== timesteps[0]:
                        h_init=height_moy
                    # print(f"for z = {z} and t = {t}, height_moy = {height_moy}")
                    row_data['height_3d_model'] = height_moy
                    row_data['Strech_3d'] = (height_moy - h_init)/ h_init
                    all_data.append(row_data)


    df = pd.DataFrame(all_data)

    # Save the DataFrame to an Excel file
    save_to_excel(df, excel_output, mode, key_columns)

# Test
# xrange = range(0, 1)
# yrange = range(0, 1)
# zrange = range(-10, 11)
# timesteps = range(0, 40)
# basepath = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_LVOT_25"
# vtk_file_pattern = r"transverse_slice_{x}_{y}_{z}.{t:02d}.vtk"
# excel_output = "center_distance_data_Z_values_LVOT_stretch.xlsx"
# mode = 'a'
# key_columns=['x', 'y', 'z', 't']
# LVOT_3D_height2excel(xrange, yrange, zrange, timesteps, basepath, vtk_file_pattern, excel_output, mode)

########################################################################################################################################################################
#%% Script gathering all data in a single excel


# Getting data from the LVOT slice
xrange = range(0, 1)
yrange = range(0, 1)
zrange = range(-10, 11)
timesteps = range(0, 40)
basepath_2d_LVOT = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25/LVOT_SSFP_CINE_25_vtk"
top_point = r"rotated_surface_1_time_{t:02d}.vtk"
bottom_point = r"rotated_surface_2_time_{t:02d}.vtk"
excel_output = "Simulation_complete_data_test.xlsx"
mode='a'
key_columns=['x', 'y', 'z', 't']
LVOT_2d_height2excel(xrange, yrange, zrange, timesteps, basepath_2d_LVOT, top_point, bottom_point, excel_output, 'w', key_columns) ### We use mode='w' here to clear the previous results
print("LVOT_2d_height2excel execution finished")

basepath_3d_LVOT = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs/sliced_CINES_LVOT_25"
vtk_file_pattern = r"transverse_slice_{x}_{y}_{z}.{t:02d}.vtk"
LVOT_3D_height2excel(xrange, yrange, zrange, timesteps, basepath_3d_LVOT, vtk_file_pattern, excel_output, mode, key_columns)
print("LVOT_3d_height2excel execution finished")

cines = [29, 30, 31, 32, 33, 34]
basepath_2dMRI = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D"
path_2d_pattern = "AO_SINUS_STACK_CINES_{cine}/AO_SINUS_STACK_CINES_{cine}_vtk"
file_2d_pattern = "rotated_{t}.vtk"
basepath_3dmodel = "C:/Users/jr403s/Documents/Model_V2_1/jobs/jobs/Whole_heart_2016_42_mesh_V3_variation/Simulation_V2/jobs"
path_3d_pattern = "sliced_CINES_{cine}"
file_3d_pattern = "transverse_slice_{x}_{y}_{z}.{t:02d}.vtk"
distance2excel_range(xrange, yrange, zrange, timesteps, cines, basepath_2dMRI, path_2d_pattern, file_2d_pattern, basepath_3dmodel, path_3d_pattern, excel_output, mode, key_columns)
print("distance2excel execution finished")

#%% calc residual : step by step script


## TEST##
timesteps = 40
cines =[29, 30, 31, 32, 33, 34]
main_path = "C:/Users/jr403s/Documents/Least_square_test/"
main_file_name = "Whole_heart_2016_42_mesh_V3_test_rewrite"
reference_file = "C:/Users/jr403s/Documents/Least_square_test/Whole_heart_2016_42_mesh_V3_reference_v1.feb"
new_simu = main_path + main_file_name 
new_feb_file = main_path + main_file_name +".feb"
new_params =  [["PrescribedDisplacement2",{'initial_value': 2, 'relative': 0} ], ["PrescribedDisplacement3",{'initial_value': -2, 'relative': 0} ], ["PrescribedDisplacement4",{'initial_value': -10, 'relative': 0} ]]
write_feb(reference_file, new_feb_file, new_params)
run_febio(main_path, main_file_name)
for t in range(timesteps):
    input_warp = new_simu + f".{t}.vtk"
    output_warp = new_simu + f"_warped.{t}.vtk"
    warp_vtk(input_warp, output_warp)
for cine in cines:
    input_seg = f"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_{cine}/AO_SINUS_STACK_CINES_{cine}_vtk"
    output_path = os.path.join(main_path, f"AO_SINUS_STACK_CINES_{cine}_vtk")
    
    model3D_pattern = main_file_name + "_warped.{t}.vtk"
    seg_pattern = "rotated_{t}.vtk"
    output_pattern = main_file_name + "_transverse_slice.{t}.vtk"
    vtk2Dslicer(main_path, input_seg, output_path, model3D_pattern, seg_pattern, output_pattern, timesteps)


path_2dMRI = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D"
path_2d_pattern = "AO_SINUS_STACK_CINES_{cine}/AO_SINUS_STACK_CINES_{cine}_vtk"
file_2d_pattern = "rotated_{t}.vtk"
basepath_3dmodel = main_path
path_3d_pattern = "AO_SINUS_STACK_CINES_{cine}_vtk"
file_3d_pattern = main_file_name + "_transverse_slice.{t}.vtk"
excel_output = main_path + "center_distance_complete_test.xlsx"
distance2excel(timesteps, cines, path_2dMRI, path_2d_pattern, file_2d_pattern, basepath_3dmodel, path_3d_pattern, file_3d_pattern, excel_output)
# %% Least square optimisation : functionnal version
from scipy.optimize import least_squares
# remember to adjust the file names to those you are using

def calc_residual(new_params):
    print(f"New params: {new_params}")
    t_systole = 11
    timesteps = 40
    cines = [29, 30, 31, 32, 33, 34]
    main_path = "C:/Users/jr403s/Documents/Least_square_test/"
    main_file_name = "Whole_heart_2016_42_mesh_V3_test_rewrite"
    reference_file = os.path.join(main_path, "Whole_heart_2016_42_mesh_V3_reference_v1.feb")

    new_simu = os.path.join(main_path, main_file_name)
    new_feb_file = os.path.join(main_path, f"{main_file_name}.feb")

    # Write the FEB file
    write_feb(reference_file, new_feb_file, new_params)

    # Run the FEBio simulation
    run_febio(main_path, main_file_name)

    # Warp VTK files
    for t in range(timesteps):
        input_warp = f"{new_simu}.{t}.vtk"
        output_warp = f"{new_simu}_warped.{t}.vtk"
        warp_vtk(input_warp, output_warp)

    # Process each cine
    for cine in cines:
        input_seg = f"C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_{cine}/AO_SINUS_STACK_CINES_{cine}_vtk"
        output_path = os.path.join(main_path, f"AO_SINUS_STACK_CINES_{cine}_vtk")

        model3D_pattern = f"{main_file_name}_warped.{{t}}.vtk"
        seg_pattern = "rotated_{t}.vtk"
        output_pattern = f"{main_file_name}_transverse_slice.{{t}}.vtk"

        vtk2Dslicer(main_path, input_seg, output_path, model3D_pattern, seg_pattern, output_pattern, timesteps)

    # Calculate distances and output to Excel
    path_2dMRI = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D"
    path_2d_pattern = "AO_SINUS_STACK_CINES_{cine}/AO_SINUS_STACK_CINES_{cine}_vtk"
    file_2d_pattern = "rotated_{t}.vtk"
    basepath_3dmodel = main_path
    path_3d_pattern = "AO_SINUS_STACK_CINES_{cine}_vtk"
    file_3d_pattern = f"{main_file_name}_transverse_slice.{{t}}.vtk"
    excel_output = os.path.join(main_path, "center_distance_complete_test.xlsx")

    df = distance2excel(timesteps, cines, path_2dMRI, path_2d_pattern, file_2d_pattern, basepath_3dmodel, path_3d_pattern, file_3d_pattern, excel_output)

    row = df[df['t'] == t_systole]

    # distances_xyz = []
    dist_x = []
    dist_y = []
    dist_z = []
    for cine in cines:
        column_name = f"distance_xyz_CINES_{cine}"
        if column_name in df.columns:
            # distances_xyz.append(row[column_name].values[0])
            dist_x.append(row[column_name].values[0][0])
            dist_y.append(row[column_name].values[0][1])
            dist_z.append(row[column_name].values[0][2])

    return dist_x, dist_y, dist_z

# Example usage
# new_params = [["PrescribedDisplacement2", {'initial_value': x, 'relative': 0}],
#               ["PrescribedDisplacement3", {'initial_value': y, 'relative': 0}],
#               ["PrescribedDisplacement4", {'initial_value': z, 'relative': 0}]]


# calc_residual(new_params)

def residuals(params):
    x, y, z = params
    new_params = [
        ["PrescribedDisplacement2", {'initial_value': x, 'relative': 0}],
        ["PrescribedDisplacement3", {'initial_value': y, 'relative': 0}],
        ["PrescribedDisplacement4", {'initial_value': z, 'relative': 0}]
    ]
    dist_x, dist_y, dist_z = calc_residual(new_params)
    # Combine the residuals from dist_x, dist_y, and dist_z
    residuals = np.concatenate([dist_x, dist_y, dist_z])
    return residuals

# print(residuals([-3, -1, -8]))
# print(residuals([0, 0, 0]))
# print(residuals([5, 5, -20]))


# Initial guess for x, y, z
initial_guess = [-3, -1., -1]

# Define bounds for each parameter
bound_range = ([-5, -3, -3], [-1, 1, 1])

# Perform the least squares optimization
result = least_squares(residuals, initial_guess, bounds = bound_range)

# Extract the optimized parameters
x_opt, y_opt, z_opt = result.x

print(f"Optimized parameters: x = {x_opt}, y = {y_opt}, z = {z_opt}")

# %%

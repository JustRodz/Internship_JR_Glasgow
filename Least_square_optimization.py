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


# %% Generic functions used for extracting data
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

#%% Calculate the misalignment distance & stretch coefficients
import os
import numpy as np
import pandas as pd

# Execute shape_center, read_points_vtk, and calculate_distances before



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



#LVOT height 2D MRI
def LVOT_2d_height2excel(timesteps, basepath, top_point_pattern, bottom_point_pattern, excel_output=None, mode='w', key_columns=['t']):
    all_data = []
    for t in range(timesteps):
        row_data = {'t': t}
        vtk_file_top_point = os.path.join(basepath, top_point_pattern.format(t=t) )
        top_point = shape_center(vtk_file_top_point)

        vtk_file_bottom_point = os.path.join(basepath, bottom_point_pattern.format(t=t))
        bottom_point = shape_center(vtk_file_bottom_point)
        
        height_t = calculate_distances_abs(top_point, bottom_point)
        if t== 0:
            h_init = height_t
        row_data['height_2d_model'] = height_t
        row_data['Strech_2d'] = (height_t - h_init)/ h_init
        all_data.append(row_data)
        # print("For t = ", t, ", the height = ", height_t)

    # Create a DataFrame with the height data
    df = pd.DataFrame(all_data)
    if excel_output is not None:
        # Save the DataFrame to an Excel file
        save_to_excel(df, excel_output, mode, key_columns)

    return(df)


#LVOT height 3D model

def LVOT_3D_height2excel(timesteps, basepath, vtk_file_pattern, excel_output=None, mode='w', key_columns=['t']):
    all_data = []
    for t in range(timesteps):
        row_data = {'t': t}
        output_pattern = vtk_file_pattern.format(t=t)
        filename = os.path.join(basepath, output_pattern)
        endpoints_test = extract_and_fit_curves(filename)
        height_loc3d = []
        for l in range(len(endpoints_test)):
            height_loc3d.append(calculate_distances_abs(endpoints_test[l][0],endpoints_test[l][1]))
        height_moy=np.mean(height_loc3d)
        if t== 0:
            h_init=height_moy
        # print(f"t = {t}, height_moy = {height_moy}")
        row_data['height_3d_model'] = height_moy
        row_data['Strech_3d'] = (height_moy - h_init)/ h_init
        all_data.append(row_data)


    df = pd.DataFrame(all_data)
    if excel_output is not None:
        # Save the DataFrame to an Excel file
        save_to_excel(df, excel_output, mode, key_columns)

    return(df)


# %% Least square optimisation : functionnal version 3 parameters
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
    
    #Calculate the misalignment distances
    df_dist = distance2excel(timesteps, cines, path_2dMRI, path_2d_pattern, file_2d_pattern, basepath_3dmodel, path_3d_pattern, file_3d_pattern, excel_output)
    row = df_dist[df_dist['t'] == t_systole]



    ## Extract the misalignment distances
    dist_x = []
    dist_y = []
    dist_z = []
    for cine in cines:
        column_name = f"distance_xyz_CINES_{cine}"
        if column_name in df_dist.columns:
            dist_x.append(row[column_name].values[0][0])
            dist_y.append(row[column_name].values[0][1])
            dist_z.append(row[column_name].values[0][2])

    ## Calculating the stretch value for the 3d LVOT
    basepath_2d_LVOT = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25/LVOT_SSFP_CINE_25_vtk"
    top_point = r"rotated_surface_1_time_{t:02d}.vtk"
    bottom_point = r"rotated_surface_2_time_{t:02d}.vtk"
    #Calculate the streching coefficients
    df_LVOT_2D = LVOT_2d_height2excel(timesteps, basepath_2d_LVOT, top_point, bottom_point)
    row_LVOT_2D = df_LVOT_2D[df_LVOT_2D['t'] == t_systole]
    column_name_2D = f"Strech_2d"
    if column_name_2D in df_LVOT_2D.columns:
        Stretch2D = row_LVOT_2D[column_name_2D].values[0]


    ## Generating the LVOT section       
    input_seg_LVOT = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25/LVOT_SSFP_CINE_25_vtk"
    output_path_LVOT = os.path.join(main_path, f"AO_SINUS_STACK_CINES_LVOT_25_vtk")
    output_pattern_LVOT = f"{main_file_name}_LVOT_slice.{{t}}.vtk"
    seg_pattern_LVOT = "rotated_surface_2_time_{t:02d}.vtk"

    vtk2Dslicer(main_path, input_seg_LVOT, output_path_LVOT, model3D_pattern, seg_pattern_LVOT, output_pattern_LVOT, timesteps)

    ## Calculating the stretch value for the 3d LVOT
    df_LVOT_3D = LVOT_3D_height2excel(timesteps, output_path_LVOT, output_pattern_LVOT)
    row_LVOT_3D = df_LVOT_3D[df_LVOT_3D['t'] == t_systole]
    column_name_3D = f'Strech_3d'
    if column_name_3D in df_LVOT_3D.columns:
        Stretch3D = row_LVOT_3D[column_name_3D].values[0]
    
    stretch_residual = [100*(Stretch3D-Stretch2D)] #Value in percent

    return dist_x, dist_y, dist_z, stretch_residual

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
    dist_x, dist_y, dist_z, stretch_residual  = calc_residual(new_params)
    print(f"Stretch residual is : {stretch_residual}")
    # Combine the residuals from dist_x, dist_y, and dist_z
    residuals = np.concatenate([dist_x, dist_y, dist_z, stretch_residual])
    return residuals

# print(residuals([-3, -1, -8]))
# print(residuals([0, 0, 0]))
# print(residuals([5, 5, -20]))


# Initial guess for x, y, z
initial_guess = [0, 0, -15]

# Define bounds for each parameter
bound_range = ([-10, -10, -30], [10, 10, 0])

# Perform the least squares optimization
result = least_squares(residuals, initial_guess, bounds = bound_range, gtol=1e-12, verbose=2)

# Extract the optimized parameters
x_opt, y_opt, z_opt = result.x

print(f"Optimized parameters: x = {x_opt}, y = {y_opt}, z = {z_opt}")


#%% Least square optimisation : functionnal version 1 parameters
from scipy.optimize import least_squares, shgo
import numpy as np
import os
import matplotlib.pyplot as plt
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
    
    #Calculate the misalignment distances
    df_dist = distance2excel(timesteps, cines, path_2dMRI, path_2d_pattern, file_2d_pattern, basepath_3dmodel, path_3d_pattern, file_3d_pattern, excel_output)
    row = df_dist[df_dist['t'] == t_systole]



    ## Extract the misalignment distances
    dist_x = []
    dist_y = []
    dist_z = []
    for cine in cines:
        column_name = f"distance_xyz_CINES_{cine}"
        if column_name in df_dist.columns:
            dist_x.append(row[column_name].values[0][0])
            dist_y.append(row[column_name].values[0][1])
            dist_z.append(row[column_name].values[0][2])

    ## Calculating the stretch value for the 3d LVOT
    basepath_2d_LVOT = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25/LVOT_SSFP_CINE_25_vtk"
    top_point = r"rotated_surface_1_time_{t:02d}.vtk"
    bottom_point = r"rotated_surface_2_time_{t:02d}.vtk"
    #Calculate the streching coefficients
    df_LVOT_2D = LVOT_2d_height2excel(timesteps, basepath_2d_LVOT, top_point, bottom_point)
    row_LVOT_2D = df_LVOT_2D[df_LVOT_2D['t'] == t_systole]
    column_name_2D = f"Strech_2d"
    if column_name_2D in df_LVOT_2D.columns:
        Stretch2D = row_LVOT_2D[column_name_2D].values[0]


    ## Generating the LVOT section       
    input_seg_LVOT = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/LVOT_seg/20160906131917_LVOT_SSFP_CINE_25/LVOT_SSFP_CINE_25_vtk"
    output_path_LVOT = os.path.join(main_path, f"AO_SINUS_STACK_CINES_LVOT_25_vtk")
    output_pattern_LVOT = f"{main_file_name}_LVOT_slice.{{t}}.vtk"
    seg_pattern_LVOT = "rotated_surface_2_time_{t:02d}.vtk"

    vtk2Dslicer(main_path, input_seg_LVOT, output_path_LVOT, model3D_pattern, seg_pattern_LVOT, output_pattern_LVOT, timesteps)

    ## Calculating the stretch value for the 3d LVOT
    df_LVOT_3D = LVOT_3D_height2excel(timesteps, output_path_LVOT, output_pattern_LVOT)
    row_LVOT_3D = df_LVOT_3D[df_LVOT_3D['t'] == t_systole]
    column_name_3D = f'Strech_3d'
    if column_name_3D in df_LVOT_3D.columns:
        Stretch3D = row_LVOT_3D[column_name_3D].values[0]
    
    stretch_residual = [100*(Stretch3D-Stretch2D)] #Value in percent

    return dist_x, dist_y, dist_z, stretch_residual

# Example usage
# new_params = [["PrescribedDisplacement2", {'initial_value': x, 'relative': 0}],
#               ["PrescribedDisplacement3", {'initial_value': y, 'relative': 0}],
#               ["PrescribedDisplacement4", {'initial_value': z, 'relative': 0}]]


# calc_residual(new_params)

def residuals(params):
    x,y = 0
    z = params
    new_params = [
        ["PrescribedDisplacement2", {'initial_value': x, 'relative': 0}],
        ["PrescribedDisplacement3", {'initial_value': y, 'relative': 0}],
        ["PrescribedDisplacement4", {'initial_value': z, 'relative': 0}]
    ]
    dist_x, dist_y, dist_z, stretch_residual  = calc_residual(new_params)
    print(f"Stretch residual is : {stretch_residual}")
    # Combine the residuals from dist_x, dist_y, and dist_z
    residuals = np.concatenate([dist_x, dist_y, dist_z, stretch_residual])
    return residuals

def objective(z):
    return np.sum(residuals(z )**2)

# zs = np.linspace(-50, 0, 50)
# print(zs)
# costs = [np.sum(residuals(z)**2) for z in zs]
# plt.plot(zs, costs)
# plt.xlabel('z')
# plt.ylabel('Somme des carrés des résidus')
# plt.show()


# Valeur initiale pour z
initial_guess = -18.4
# Définir les bornes uniquement pour z
bounds = (-30, -10)
# Effectuer l'optimisation uniquement sur z
result_ls = least_squares(residuals, initial_guess, bounds=bounds, gtol=5e-16, verbose=2)
# Extraire le paramètre optimisé
print(result_ls)
z_opt_ls = result_ls.x[0]
print(f"Optimized parameter: z = {z_opt_ls}")

# bounds_shgo = [(-18, -16)]
# result_shgo = shgo(objective, bounds=bounds_shgo, n=20, iters=1, sampling_method='sobol', options={'disp': True})
# print(result_shgo)
# z_opt_shgo = result_shgo.x[0]
# print(f"Optimized parameter (shgo): z = {z_opt_shgo}")
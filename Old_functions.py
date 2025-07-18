### All python script here are unused/previous part of the Combined_Functions.py script
### None of them remain in the current version of Combined_Functions.py
### This file main use is in case we happen to need any of those script once again

#%% Levelset segmentation
#%% Strategy to extract the contour of a segmentation 
#2 Steps : Improve the segmentation using a Levelset filter (nifti) + extract the contour of the segmentation (.vtk)
def apply_levelset_seg_simpleitk(nifti_image_path, nifti_mask_path, nifti_output_path=None, vtk_output_path=None, iterations=50):
    """
    Args:
        nifti_image_path (str): Path to original image (.nii or .nii.gz)
        nifti_mask_path (str): Path to the initial segmentation (binary) in NIfTI format
        nifti_output_path (str, optional): Path to save the result (NIfTI). If None, do not save.
        vtk_output_path (str, optional): Path to save the point cloud (VTK). If None, do not save.
        iterations (int): Number of iterations of the level-set evolution.
    """

    # Load data with SimpleITK
    img_sitk = sitk.ReadImage(nifti_image_path, sitk.sitkFloat32)
    mask_sitk = sitk.ReadImage(nifti_mask_path, sitk.sitkFloat32)

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
            filter.SetPropagationScaling(0.05)
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
    if nifti_output_path:
        sitk.WriteImage(result_sitk, nifti_output_path)

    # Export the contour as a VTK point cloud
    if vtk_output_path:
        for t in range(num_timesteps):
            # Extract the current timestep
            segmented_timestep = segmented_timesteps[t]

            # Convert the segmented timestep to a numpy array
            segmented_array = sitk.GetArrayFromImage(segmented_timestep)

            # Check if the data is 2D or 3D
            if len(segmented_array.shape) == 3:
                # Determine the plane for 2D slice extraction
                depth, height, width = segmented_array.shape
                if depth == 1:
                    segmented_slice = segmented_array[0, :, :]
                elif height == 1:
                    segmented_slice = segmented_array[:, 0, :]
                elif width == 1:
                    segmented_slice = segmented_array[:, :, 0]
                else:
                    # Default to the first slice along the first dimension if no single slice dimension is found
                    segmented_slice = segmented_array[0, :, :]
            else:
                segmented_slice = segmented_array

            # Normalize the slice to the range [0, 255] and convert to uint8
            segmented_slice_normalized = ((segmented_slice - segmented_slice.min()) / (segmented_slice.max() - segmented_slice.min()) * 255).astype(np.uint8)

            # # Debug: Visualize the segmented slice
            # plt.imshow(segmented_slice_normalized, cmap='gray')
            # plt.title("Segmented Array Slice")
            # plt.show()

            # Apply Canny edge detection to find the contour
            edges = cv2.Canny(segmented_slice_normalized, 50, 150)  # Adjust thresholds as needed

            # # Debug : Visualize the edges
            # plt.imshow(edges, cmap='gray')
            # plt.title("Edges")
            # plt.show()


            # Get the coordinates of the edge points
            points = np.argwhere(edges > 0)

            vtk_points = vtk.vtkPoints()
            for point in points:
                if point.shape[0] == 2:
                    # Insert the point into the VTK points object with z=0 for 2D points
                    vtk_points.InsertNextPoint(point[1], point[0], 0)
                else:
                    # Insert the point into the VTK points object for 3D points
                    vtk_points.InsertNextPoint(point[2], point[1], point[0])

            # Create a VTK polydata object and set the points
            vtk_path = os.path.join(vtk_output_path, f"contour_{t}.vtk")
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_points)

            # Write the polydata to a VTK file
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(vtk_path)
            writer.SetInputData(polydata)
            writer.Write()

    return result_sitk

# Test
affine = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/20160906131917_AO_SINUS_STACK_CINES_29.nii"
mask = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/20160906131917_AO_SINUS_STACK_CINES_29_Segmentation.nii"
nifti_output_path = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/20160906131917_AO_SINUS_STACK_CINES_29_Seg_levelset.nii"
vtk_output_path = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/"
seg = apply_levelset_seg_simpleitk(affine, mask, nifti_output_path, vtk_output_path, iterations=25)
print("Level set done")


# %% Reordering points if we have a non continuous outline
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


#%% Recenter points
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
def are_coplanar(points):
    if len(points) < 3:  #Pasaasez de points pour obtenir un plan
        return False

    v1 = points[1] - points[0]
    v2 = points[2] - points[0]   #Two points to points vector
    normal_vector = np.cross(v1, v2)

    for point in points[3:]:
        if not np.isclose(np.dot(normal_vector, point - points[0]), 0, 0.001): # initialement if np.dot(normal_vector, point - points[0])== 0, mais isclose permet de gerer des approximations
            return False
    return True

#%% Optimization using Kabsch algorithm
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
    # B_aligned=reorder_points(B_aligned) ################################################## test
    return B_aligned


def project_to_plane(points, normal, point_on_plane):
    """
    Projects 3D points onto a plane defined by a normal vector and a point on the plane.
    """
    normal = normal / np.linalg.norm(normal)
    vectors = points - point_on_plane
    distances = np.dot(vectors, normal)
    return points - np.outer(distances, normal)



##Test~~~~~

input_path_A = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/rotated_4.vtk"
matrice_A = recentrer_points(input_path_A)
matrice_A = reorder_points(matrice_A)
input_path_B = "C:/Users/jr403s/Documents/Test_segmentation_itk/Segmentation_2D/AO_SINUS_STACK_CINES_29/AO_SINUS_STACK_CINES_29_vtk/transverse_slice_004.vtk"
matrice_B = recentrer_points(input_path_B)
matrice_B = reorder_points(matrice_B)

# Interpolate the matrix
new_number_points = min(matrice_A.shape[0], matrice_B.shape[0])
matrice_A_interpolee = interpolate_along_curve(matrice_A, new_number_points)
matrice_B_interpolee = interpolate_along_curve(matrice_B, new_number_points)



############# Optimisation

# Calculate the original residual gap
gap_init = gap_calculator_array(matrice_A_interpolee, matrice_B_interpolee)
print("Gap before optimization :", gap_init)

matrice_A_opti = kabsch_align(matrice_B_interpolee, matrice_A_interpolee)

# # Reprojeter dans le plan de A (Ensure the optimised result is in the same plan as matrice_A)
# plane_normal = np.cross(matrice_A_interpolee[1] - matrice_A_interpolee[0], matrice_A_interpolee[2] - matrice_A_interpolee[0])
# plane_point = matrice_A_interpolee[0]
# matrice_B_opti = project_to_plane(matrice_B_opti, plane_normal, plane_point)

# Calculate the final residual gap
gap_final = gap_calculator_array(matrice_B_interpolee, matrice_A_opti)
print("Gap after optimization :", gap_final)



##### Visualization

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

# # Courbe B optimisée (vert)
# fig.add_trace(go.Scatter3d(
#     x=matrice_A_opti[:, 0], y=matrice_A_opti[:, 1], z=matrice_A_opti[:, 2],
#     mode='lines+markers',
#     name='Optimised A',
#     line=dict(color='green', dash='dash'),
#     marker=dict(size=3, color='green')
# ))

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
    height=600
)

fig.show()

#%%
# Python tool  aiming to modify the boundaries of the febio model
def modify_feb_file(input_file_path, output_file_path, new_boundary_conditions):
    """
    Args:
    input_file path (str): Path to original febio file (.feb)
    output_file_path (str): Path where the new .feb file with updated boundaries will be saved
    new_boundary_conditions (str): Listing of the new boundaries. Written to be compatible with febio
    """
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    with open(output_file_path, 'w') as file:
        in_boundary_condition_section = False

        for line in lines:
            if line.strip() == "<BoundaryCondition>" or line.strip() == "<Boundary>":
                in_boundary_condition_section = True
                file.write(line)
                # Write new boundary conditions
                for condition in new_boundary_conditions:
                    file.write(f'    <fix id="{condition["id"]}">{condition["values"]}</fix>\n')
            elif line.strip() == "</BoundaryCondition>" or line.strip() == "</Boundary>":
                in_boundary_condition_section = False
                file.write(line)
            elif not in_boundary_condition_section:
                file.write(line)

# Example
input_file_path = "C:/Users/jr403s/Documents/Model_V2_1/jobs/Whole_heart_2016_42_mesh_V2_PreSim.feb"
output_file_path = "C:/Users/jr403s/Documents/Model_V2_1/jobs/Whole_heart_2016_42_mesh_V2_PreSim _modified_BC_CrashTesting.feb"
new_boundary_conditions =  [
    {"content": '<bc name="ZeroDisplacement1" node_set="@edge:ZeroDisplacement1" type="zero displacement"><x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof></bc>'},
    {"content": '<bc name="PrescribedDisplacement2" node_set="@edge:PrescribedDisplacement2" type="prescribed displacement"><dof>x</dof><value lc="3">1</value><relative>0</relative></bc>'},
    {"content": '<bc name="PrescribedDisplacement3" node_set="@edge:PrescribedDisplacement3" type="prescribed displacement"><dof>y</dof><value lc="3">0</value><relative>0</relative></bc>'},
    {"content": '<bc name="PrescribedDisplacement4" node_set="@edge:PrescribedDisplacement4" type="prescribed displacement"><dof>z</dof><value lc="3">0</value><relative>0</relative></bc>'}
]
    # Add other boundary conditions if necessary

modify_feb_file(input_file_path, output_file_path, new_boundary_conditions)
print("modification done")



# %% Regression multiple
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Charger les données
file_path = "C:/Users/jr403s/Internship_JR_Glasgow/code/center_distance_data_V3_test.xlsx"
data = pd.read_excel(file_path)

# Filtrer les données pour t = 15
t_constant_data = data[data['t'] == 15].dropna()

# Définir les variables indépendantes et dépendantes
X = t_constant_data[['i', 'j', 'k']]
y = t_constant_data['distance_CINES_29']

# Ajouter une constante pour l'interception
X = sm.add_constant(X)

# Ajuster le modèle de régression multiple
model = sm.OLS(y, X).fit()

# Résumé du modèle
print(model.summary())

# Calculer le VIF pour chaque variable indépendante
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)

# %% Regression polynomiale
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# access data
file_path =  "C:/Users/jr403s/Internship_JR_Glasgow/code/center_distance_data_V3_test.xlsx"
data = pd.read_excel(file_path)

# Filtrer les données pour t = 15 fixe
t_constant_data = data[data['t'] == 15].dropna()

# Définir les variables indépendantes et dépendantes
X = t_constant_data[['i', 'j', 'k']]
y = t_constant_data['distance_CINES_29']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Ajuster le modèle de régression polynomiale
model = LinearRegression()
model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("------ Regression polynomiale ------")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# %% Plot distance_CINES_XX/xyz_norm
import pandas as pd
import numpy as np
import plotly.express as px

# Load the Excel file
file_path = "C:/Users/jr403s/Documents/code/center_distance_data_V3.xlsx"
data = pd.read_excel(file_path)

# Filter data for t = 15
t_constant_data = data[data['t'] == 15].copy()

# Calculate the norm of the vectors (i, j, k)
t_constant_data['norm'] = np.sqrt(t_constant_data['i']**2 + t_constant_data['j']**2 + t_constant_data['k']**2)
for cine in [29, 30, 31, 32, 33, 34]:
    # Calculate mean and standard deviation for each norm value
    stats = t_constant_data.groupby('norm')[f'distance_CINES_{cine}'].agg(['mean', 'std']).reset_index()

    # Plot the relationship between Norm and distance_CINES_<cine>
    fig1 = px.scatter(t_constant_data, x='norm', y=f"distance_CINES_{cine}",
                     title=f"Relation entre la Norme et la Distance CINES {cine}",
                     labels={'norm': 'Norme', f"distance_CINES_{cine}": f"Distance CINES {cine}"},
                     trendline="ols")  # Add linear trendline

   
    
    # Add standard deviation to the plot
    fig1.add_trace(go.Scatter(
        x=stats['norm'],
        y=stats['mean'] + stats['std'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig1.add_trace(go.Scatter(
        x=stats['norm'],
        y=stats['mean']- stats['std'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    ))


    ##########################################################
    # Plot the number of combinations for each norm value
    fig2 = px.bar(norm_counts, x='norm', y='count',
                title='Nombre de combinaisons (i, j, k) par Norme',
                labels={'norm': 'Norme', 'count': 'Nombre de combinaisons'})

    # Show plots3433
    fig1.show(), fig2.show()


#%% Point tarcker for LVOT (Associating key points from the MRI 2D segmentation to the 3D model)

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


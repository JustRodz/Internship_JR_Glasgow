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
# %%


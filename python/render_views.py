import laspy
import numpy as np
import open3d as o3d
import os

def read_las_file(file_path):
    """Reads a .las file and returns a numpy array of point coordinates, centered at the origin."""
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    center = points.mean(axis=0)
    points -= center  # Shift points to be centered at the origin
    return points

def render_views(points, output_dir):
    """Renders and saves 8 top-down views and 2 section views."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute bounding box center (now should be near (0,0,0))
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    center = (min_vals + max_vals) / 2
    
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 1.0  # Decrease point size
    ctr = vis.get_view_control()
    
        
    # Render top-down views from 8 angles
    for i, angle in enumerate(range(0, 360, 15)):
        ctr.set_front([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0.5])  # Ensure Z-axis is pointing upwards
        ctr.set_lookat(center)
        ctr.set_up([0, 0, 1])  # Up direction is strictly Z
        #ctr.set_zoom(0.5)
        vis.poll_events()
        vis.update_renderer()
        output_path = os.path.join(output_dir, f'top_view_{i:02d}.png')
        vis.capture_screen_image(output_path)
        
    
    # Render section views
    section_width = 5  # 5m wide sections
    mask_ns = (points[:, 0] > center[0] - section_width / 2) & (points[:, 0] < center[0] + section_width / 2)
    mask_ew = (points[:, 1] > center[1] - section_width / 2) & (points[:, 1] < center[1] + section_width / 2)
    
    for direction, mask, front in zip(['ns', 'ew'], [mask_ns, mask_ew], [[1, 0, 0], [0, -1, 0]]):
        section_points = points[mask]
        section_pcd = o3d.geometry.PointCloud()
        section_pcd.points = o3d.utility.Vector3dVector(section_points)
        vis.clear_geometries()
        vis.add_geometry(section_pcd)
        ctr.set_front(front)  # Set correct viewing direction
        ctr.set_lookat(center)
        ctr.set_up([0, 0, 1])  # Keep Z-axis as up
        ctr.set_zoom(0.5)
        vis.poll_events()
        vis.update_renderer()
        output_path = os.path.join(output_dir, f'section_{direction}.png')
        vis.capture_screen_image(output_path)
    
    vis.destroy_window()


# Example usage:
points = read_las_file(r'E:\Ecosense\2024-10-15 ecosense.RiSCAN\EXPORTS\Export Point Clouds\circles\segmentation_circle_2.las')
render_views(points, r'E:\Ecosense\2024-10-15 ecosense.RiSCAN\EXPORTS\Export Point Clouds\circles\circ1')

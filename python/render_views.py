import laspy
import numpy as np
import open3d as o3d
import os
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Transformer, CRS

def read_las_file(file_path):
    """Reads a .las file and returns a numpy array of point coordinates, centered at the origin, 
    along with the bounding box as a spatial polygon in EPSG:3857."""
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    center = points.mean(axis=0)
    
    # Compute bounding box in original coordinates
    min_x, min_y = np.min(points[:, :2], axis=0)
    max_x, max_y = np.max(points[:, :2], axis=0)
    
    points -= center  # Shift points to be centered at the origin

    
    # Define bounding box polygon
    bbox_polygon = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
    
    # Get CRS from LAS file
    input_crs = None
    if las.header.parse_crs():
        input_crs = las.header.parse_crs()
    if input_crs is None:
        raise ValueError("No spatial reference found in LAS file.")
    
    # Transform to EPSG:3857 (Web Mercator)
    transformer = Transformer.from_crs(input_crs, "EPSG:3857", always_xy=True)
    transformed_coords = [transformer.transform(x, y) for x, y in bbox_polygon.exterior.coords]
    bbox_polygon_3857 = Polygon(transformed_coords)
    
    return points, bbox_polygon_3857

def save_bounding_box_as_gpkg(bbox_polygon, output_path):
    """Saves the bounding box polygon as a GeoPackage file."""
    gdf = gpd.GeoDataFrame(geometry=[bbox_polygon], crs="EPSG:3857")
    gdf.to_file(output_path, driver="GPKG")

def render_views(points, output_dir, max_points=1e8):
    """Renders and saves 8 top-down views and 2 section views."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute bounding box center (now should be near (0,0,0))
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    center = (min_vals + max_vals) / 2
    
    # get sections 
    section_width = 5  # 5m wide sections
    mask_ns = (points[:, 0] > center[0] - section_width / 2) & (points[:, 0] < center[0] + section_width / 2)
    mask_ew = (points[:, 1] > center[1] - section_width / 2) & (points[:, 1] < center[1] + section_width / 2)
    
    # Downsample and convert to Open3D point cloud
    if(len(points) > max_points):
        np.random.shuffle(points)
        points = points[:int(max_points)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    
    
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
points, polygon = read_las_file(r'E:\Ecosense\2024-10-15 ecosense.RiSCAN\EXPORTS\Export Point Clouds\2024-10-15 ecosense 0.020 m.las')
save_bounding_box_as_gpkg(polygon, r'E:\Ecosense\2024-10-15 ecosense.RiSCAN\EXPORTS\Export Point Clouds\render\bounding_box.gpkg')
render_views(points, r'E:\Ecosense\2024-10-15 ecosense.RiSCAN\EXPORTS\Export Point Clouds\render')

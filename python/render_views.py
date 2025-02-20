import laspy
import numpy as np
import open3d as o3d
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import argparse
import json
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
    
    # Define bounding box polygon
    bbox_polygon = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

    points -= center  # Shift points to be centered at the origin
  
    # Get CRS from LAS file
    input_crs = None
    if las.header.parse_crs():
        input_crs = las.header.parse_crs()
    if input_crs is None:
        raise ValueError("No spatial reference found in LAS file.")
    
    # Transform to EPSG:3857 (Web Mercator)
    transformer = Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)
    transformed_coords = [transformer.transform(x, y) for x, y in bbox_polygon.exterior.coords]
    bbox_polygon_4326 = Polygon(transformed_coords)
    
    return points, bbox_polygon_4326

def save_bounding_box_as_geojson(bbox_polygon, output_path, metadata):
    """Saves the bounding box polygon as a GeoJSON file, embedding metadata."""
    gdf = gpd.GeoDataFrame([metadata], geometry=[bbox_polygon], crs="EPSG:3857")
    gdf.to_file(output_path, driver="GeoJSON")

def render_views(points, output_dir, max_points, section_width, image_width, image_height):
    """Renders and saves 8 top-down views and 2 section views."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute bounding box center
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    center = (min_vals + max_vals) / 2

    # Generate mask for random downsampling
    if len(points) > max_points:
        mask_random = np.random.choice(len(points), int(max_points), replace=False)
    else:
        mask_random = np.arange(len(points))
    
    sampled_points = points[mask_random]
    
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    # Apply Viridis colormap for elevation
    heights = sampled_points[:, 2]
    colors = plt.get_cmap("viridis")((heights - heights.min()) / (heights.max() - heights.min()))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=image_width, height=image_height)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 1.0  # Decrease point size
    ctr = vis.get_view_control()
    
    # Render top-down views from 8 angles
    for i, angle in enumerate(range(0, 360, 10)):
        ctr.set_front([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0.5])
        ctr.set_lookat(center)
        ctr.set_up([0, 0, 1])
        vis.poll_events()
        vis.update_renderer()
        output_path = os.path.join(output_dir, f'top_view_{i:02d}.png')
        vis.capture_screen_image(output_path)
    
    # Render section views
    mask_ns = (points[:, 0] > center[0] - section_width / 2) & (points[:, 0] < center[0] + section_width / 2)
    mask_ew = (points[:, 1] > center[1] - section_width / 2) & (points[:, 1] < center[1] + section_width / 2)
    
    for direction, mask, front in zip(['ns', 'ew'], [mask_ns, mask_ew], [[1, 0, 0], [0, 1, 0]]):
        section_points = points[mask]
        section_pcd = o3d.geometry.PointCloud()
        section_pcd.points = o3d.utility.Vector3dVector(section_points)
        
        # Apply Viridis colormap along the viewing axis
        color_values = section_points[:, 1] if direction == 'ew' else section_points[:, 0]
        colors = plt.get_cmap("viridis")((color_values - color_values.min()) / (color_values.max() - color_values.min()))[:, :3]
        section_pcd.colors = o3d.utility.Vector3dVector(colors)
        
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

def main():
    parser = argparse.ArgumentParser(description="Render LiDAR point clouds with Open3D.")
    parser.add_argument("input_file", type=str, help="Path to the input LAS file.")
    parser.add_argument("output_folder", type=str, help="Path to the output directory.")
    parser.add_argument("--max_points", type=int, default=int(1e8), help="Maximum number of points to render.")
    parser.add_argument("--section_width", type=float, default=10, help="Width of section views in meters.")
    parser.add_argument("--image_width", type=int, default=2048, help="Height of the output images in pixels.")
    parser.add_argument("--image_height", type=int, default=1152, help="Width of the output images in pixels.")
    
    args = parser.parse_args()
    
    # Load LAS file and extract bounding box
    points, bbox_polygon = read_las_file(args.input_file)
    
    # Save bounding box as GeoJSON with metadata
    metadata = {
        "input_file": args.input_file,
        "output_folder": args.output_folder,
        "max_points": args.max_points,
        "section_width": args.section_width,
        "image_width": args.image_width,
        "image_height": args.image_height
    }
    bbox_path = os.path.join(args.output_folder, "bounding_box.geojson")
    save_bounding_box_as_geojson(bbox_polygon, bbox_path, metadata)
    
    # Render views
    render_views(points, args.output_folder, args.max_points, args.section_width, args.image_width, args.image_height)
    
if __name__ == "__main__":
    main()

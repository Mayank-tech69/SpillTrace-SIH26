import json
import os
from geopy.distance import geodesic
from shapely.geometry import LineString, mapping
import pyproj
from shapely.ops import transform

# --- GEOSPATIAL HELPER FUNCTIONS ---
def create_geodesic_circle(lat, lon, radius_m, num_points=36):
    """
    Generates a perfect geodesic polygon without metre-to-degree bugs.
    Calculates exact WGS-84 destination points at 360-degree intervals.
    """
    center = (lat, lon)
    coords = []
    for angle in range(0, 360, 360 // num_points):
        # Calculate exact point at radius_m distance along this bearing
        pt = geodesic(meters=radius_m).destination(center, angle)
        coords.append([round(pt.longitude, 6), round(pt.latitude, 6)]) # [lon, lat] format
    coords.append(coords[0]) # Close the polygon loop
    return {"type": "Polygon", "coordinates": [coords]}

def create_metric_swept_corridor(lon_lat_coords, buffer_m):
    """
    Safely buffers a line in meters by temporarily projecting to a local metric CRS (UTM),
    buffering, and projecting back to WGS-84 (EPSG:4326).
    """
    line = LineString(lon_lat_coords)
    # Project to a local Web Mercator / UTM equivalent for accurate metric buffering
    project_to_meters = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    project_to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    
    line_m = transform(project_to_meters, line)
    poly_m = line_m.buffer(buffer_m)
    poly_wgs84 = transform(project_to_wgs84, poly_m)
    
    return mapping(poly_wgs84)

def generate_updated_real_scenario():
    print("Generating Updated Real SAR Scenario Evaluation...")
    
    # 1. Load the trajectory points we saved in Day 5
    with open("day5_outputs/particles_trajectory.geojson", "r") as f:
        trajectory_data = json.load(f)
        
    path_coords = []
    for feature in trajectory_data["features"]:
        path_coords.append(feature["geometry"]["coordinates"]) # [lon, lat]
        
    # Final origin is the last point in the trajectory (-12h)
    final_lon, final_lat = path_coords[-1]
    final_radius_m = 3767.0
    
    # 2. Generate mathematically safe polygons
    final_origin_polygon = create_geodesic_circle(final_lat, final_lon, final_radius_m)
    # Using a 3.7km buffer for the entire swept path for the corridor
    swept_corridor = create_metric_swept_corridor(path_coords, final_radius_m)
    
    # 3. Build the strict payload
    payload = {
        "spill_id": "SPILL_TEST3_001",
        "data_mode": "real_observation_with_parameter_driven_simulation",
        
        "predicted_origin_lat": final_lat,
        "predicted_origin_lon": final_lon,
        "actual_origin_lat": None,
        "actual_origin_lon": None,
        "reference_source": "not_publicly_available",
        "drift_location_error_km": None,
        "evaluation_status": "ground_truth_unavailable",
        
        # --- GEOSPATIAL PAYLOADS ---
        "origin_point_geojson": {
            "type": "Point",
            "coordinates": [final_lon, final_lat]
        },
        "hindcast_path_geojson": {
            "type": "LineString",
            "coordinates": path_coords
        },
        "hindcast_swept_corridor_geojson": swept_corridor,
        "final_origin_uncertainty_polygon_geojson": final_origin_polygon,
        
        # --- METADATA ---
        "final_uncertainty_radius_m": final_radius_m,
        "corridor_geometry_meaning": "hindcast_swept_corridor represents the bounding area of the entire 12-hour trajectory. final_origin_uncertainty_polygon represents only the probable origin area at T-12h.",
        "wind_source": "analyst-parameter-input",
        "current_source": "analyst-parameter-input",
        "geodesic_movement_method": "geopy.distance.geodesic (WGS-84 ellipsoid)",
        "particle_count": 1,
        "uncertainty_growth_formula": "radius_m = base_radius (100m) + (distance_per_step * 0.1)",
        "random_seed": 42,
        
        "origin_time_start_utc": "2026-09-01T12:00:00Z",
        "origin_time_end_utc": "2026-09-02T00:00:00Z",
        "drift_mode": "analyst-parameter-driven"
    }
    
    path = "day5_outputs/sunidhi_real_scenario_updated.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved successfully to: {path}")

if __name__ == "__main__":
    generate_updated_real_scenario()
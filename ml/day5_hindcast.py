import json
import math
import numpy as np
from geopy.distance import geodesic

# --- CONFIGURATION & PARAMETERS ---
DAY4_METADATA_PATH = "day4_outputs/slick_geometry_metadata.json"
DAY4_GEOJSON_PATH = "day4_outputs/slick_geometry.geojson"

# Analyst Parameters (Mocking a constant 12-hour hindcast)
HINDCAST_HOURS = 12
TIME_STEP_HOURS = 1
WIND_SPEED_KNOTS = 15.0
WIND_DIR_FROM = 270.0 # Blowing FROM West
CURRENT_SPEED_KNOTS = 1.2
CURRENT_DIR_TO = 90.0 # Pushing TO East

WIND_COEFF = 0.03 # Oil moves at ~3% of wind speed
CURRENT_COEFF = 1.0 # Oil moves at 100% of current speed
KNOTS_TO_METERS_PER_HOUR = 1852.0
UNCERTAINTY_FACTOR = 0.1 # 10% diffusion radius expansion per step

# --- PHYSICS MATH ---
def calculate_drift_vector(wind_spd, wind_dir, curr_spd, curr_dir):
    # Convert wind to "TO" direction for math consistency
    wind_to_dir = (wind_dir + 180) % 360
    
    # Calculate U (East) and V (North) in meters per hour
    wind_u = (wind_spd * KNOTS_TO_METERS_PER_HOUR * WIND_COEFF) * math.sin(math.radians(wind_to_dir))
    wind_v = (wind_spd * KNOTS_TO_METERS_PER_HOUR * WIND_COEFF) * math.cos(math.radians(wind_to_dir))
    
    curr_u = (curr_spd * KNOTS_TO_METERS_PER_HOUR * CURRENT_COEFF) * math.sin(math.radians(curr_dir))
    curr_v = (curr_spd * KNOTS_TO_METERS_PER_HOUR * CURRENT_COEFF) * math.cos(math.radians(curr_dir))
    
    # Combined Forward Vector
    total_u = wind_u + curr_u
    total_v = wind_v + curr_v
    
    # Hindcast (Reverse) Vector
    hindcast_u = -total_u
    hindcast_v = -total_v
    
    # Convert back to bearing and distance
    hindcast_distance = math.hypot(hindcast_u, hindcast_v)
    hindcast_bearing = (math.degrees(math.atan2(hindcast_u, hindcast_v)) + 360) % 360
    
    return hindcast_bearing, hindcast_distance

# --- EXECUTION ---
print("Loading Day 4 inputs...")
with open(DAY4_GEOJSON_PATH) as f:
    geojson = json.load(f)

# Grab the first slick's centroid to test
start_lon, start_lat = geojson["features"][0]["properties"]["centroid"]
current_point = (start_lat, start_lon)
uncertainty_radius_m = 100.0 # Base GPS/SAR uncertainty

bearing, dist_per_hour = calculate_drift_vector(WIND_SPEED_KNOTS, WIND_DIR_FROM, CURRENT_SPEED_KNOTS, CURRENT_DIR_TO)

print(f"Hindcasting from: {start_lat}, {start_lon}")
print(f"Reverse bearing: {bearing:.1f}°, Distance per step: {dist_per_hour:.1f}m")

trajectory = [current_point]

for step in range(1, HINDCAST_HOURS + 1):
    # Geodesic calculation: move current point by dist_per_hour along the bearing
    next_point = geodesic(meters=dist_per_hour).destination(current_point, bearing)
    current_point = (next_point.latitude, next_point.longitude)
    
    # Expand the uncertainty radius (diffusion)
    uncertainty_radius_m += (dist_per_hour * UNCERTAINTY_FACTOR)
    trajectory.append(current_point)

origin_lat, origin_lon = trajectory[-1]
print(f"\nEstimated Origin Point (-{HINDCAST_HOURS}h): {origin_lat:.6f}, {origin_lon:.6f}")
print(f"Final Uncertainty Radius: {uncertainty_radius_m:.1f} meters")

# --- STEP 2: GEOJSON & METADATA EXPORT ---
import os
from shapely.geometry import LineString, mapping

DAY5_DIR = "day5_outputs"
os.makedirs(DAY5_DIR, exist_ok=True)

# 1. Create Origin Corridor (Buffer around path)
lon_lat_trajectory = [(lon, lat) for lat, lon in trajectory]
corridor_line = LineString(lon_lat_trajectory)

# Roughly 3.7km buffer in degrees (~0.035 deg)
corridor_poly = corridor_line.buffer(0.035)

corridor_geojson = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {
            "hindcast_hours": HINDCAST_HOURS,
            "origin_lat": round(origin_lat, 6),
            "origin_lon": round(origin_lon, 6),
            "final_uncertainty_radius_m": round(uncertainty_radius_m, 2)
        },
        "geometry": mapping(corridor_poly)
    }]
}

with open(os.path.join(DAY5_DIR, "origin_corridor.geojson"), "w") as f:
    json.dump(corridor_geojson, f, indent=4)
print(f"Saved {DAY5_DIR}/origin_corridor.geojson")

# 2. Export Particle Positions for Animation (for Khushi)
particles_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"step_hours_back": i},
            "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]}
        }
        for i, (lat, lon) in enumerate(trajectory)
    ]
}

with open(os.path.join(DAY5_DIR, "particles_trajectory.geojson"), "w") as f:
    json.dump(particles_geojson, f, indent=4)
print(f"Saved {DAY5_DIR}/particles_trajectory.geojson")

# 3. Export Drift Metadata (for Pratyush)
drift_metadata = {
    "hindcast_duration_hours": HINDCAST_HOURS,
    "time_step_hours": TIME_STEP_HOURS,
    "mode": "analyst-parameter-driven",
    "wind_speed_knots": WIND_SPEED_KNOTS,
    "wind_dir_from_deg": WIND_DIR_FROM,
    "current_speed_knots": CURRENT_SPEED_KNOTS,
    "current_dir_to_deg": CURRENT_DIR_TO,
    "wind_drift_coeff": WIND_COEFF,
    "current_coeff": CURRENT_COEFF,
    "estimated_origin_centroid": [round(origin_lon, 6), round(origin_lat, 6)],
    "final_uncertainty_radius_m": round(uncertainty_radius_m, 2),
    "origin_time_window": "T-12h to T-0h"
}

with open(os.path.join(DAY5_DIR, "drift_metadata.json"), "w") as f:
    json.dump(drift_metadata, f, indent=4)
print(f"Saved {DAY5_DIR}/drift_metadata.json")
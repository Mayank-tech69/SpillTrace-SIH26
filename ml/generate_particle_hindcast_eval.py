import json
import math
import os
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import MultiPoint, LineString, mapping
import pyproj
from shapely.ops import transform

# --- CONFIGURATION & REPRODUCIBILITY ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PARTICLE_COUNT = 250
DURATION_HOURS = 12
TIMESTEP_SECONDS = 3600
TIMESTEP_HOURS = 1.0

# Observed Slick Centroid (from Day 4)
OBS_LAT = 19.495814
OBS_LON = 71.058396

# Physical Forcing Parameters (Strict SI units)
WIND_SPEED_MPS = 7.7167       # ~15.0 knots
WIND_DIR_FROM_DEG = 270.0     # Wind blowing FROM West (270°)
CURRENT_SPEED_MPS = 0.6173     # ~1.2 knots
CURRENT_DIR_TO_DEG = 90.0      # Current pushing TO East (90°)

WIND_DRIFT_COEFF = 0.03
CURRENT_COEFF = 1.0

# Turbulent Diffusion Coefficient (m^2/s) - typical ocean surface value
DIFFUSIVITY_M2_S = 1.0
DIFFUSION_STEP_SIGMA_M = math.sqrt(2.0 * DIFFUSIVITY_M2_S * TIMESTEP_SECONDS) # ~84.85 m/step

# --- VECTOR KINEMATICS ---
def get_base_advection_step():
    # Convert wind FROM to TO
    wind_dir_to = (WIND_DIR_FROM_DEG + 180.0) % 360.0
    
    wind_u = WIND_SPEED_MPS * WIND_DRIFT_COEFF * math.sin(math.radians(wind_dir_to))
    wind_v = WIND_SPEED_MPS * WIND_DRIFT_COEFF * math.cos(math.radians(wind_dir_to))
    curr_u = CURRENT_SPEED_MPS * CURRENT_COEFF * math.sin(math.radians(CURRENT_DIR_TO_DEG))
    curr_v = CURRENT_SPEED_MPS * CURRENT_COEFF * math.cos(math.radians(CURRENT_DIR_TO_DEG))
    
    # Combined hindcast (reverse) displacement per second
    hindcast_u = -(wind_u + curr_u)
    hindcast_v = -(wind_v + curr_v)
    
    # Step distance in meters per hour
    step_distance_m = math.hypot(hindcast_u, hindcast_v) * TIMESTEP_SECONDS
    step_bearing = (math.degrees(math.atan2(hindcast_u, hindcast_v)) + 360.0) % 360.0
    
    return step_bearing, step_distance_m

def run_monte_carlo_hindcast():
    print(f"Running Monte Carlo Hindcast ({PARTICLE_COUNT} particles, seed={RANDOM_SEED})...")
    base_bearing, base_step_dist_m = get_base_advection_step()
    
    # Initial particle perturbations (~25m initial slick radius)
    initial_offsets_m = np.random.normal(0, 25.0, (PARTICLE_COUNT, 2))
    particles = []
    
    for i in range(PARTICLE_COUNT):
        p_pt = (OBS_LAT, OBS_LON)
        dx, dy = initial_offsets_m[i]
        init_dist = math.hypot(dx, dy)
        if init_dist > 0:
            init_bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
            p_pt = geodesic(meters=init_dist).destination(p_pt, init_bearing)[:2]
        particles.append([p_pt])
    
    # Advect particles backward across time steps
    for step in range(DURATION_HOURS):
        for i in range(PARTICLE_COUNT):
            current_lat, current_lon = particles[i][-1]
            
            # 1. Base physical drift
            advected = geodesic(meters=base_step_dist_m).destination((current_lat, current_lon), base_bearing)[:2]
            
            # 2. Stochastic Brownian diffusion perturbation
            diff_x = np.random.normal(0, DIFFUSION_STEP_SIGMA_M)
            diff_y = np.random.normal(0, DIFFUSION_STEP_SIGMA_M)
            diff_dist = math.hypot(diff_x, diff_y)
            diff_bearing = (math.degrees(math.atan2(diff_x, diff_y)) + 360.0) % 360.0
            
            diffused = geodesic(meters=diff_dist).destination(advected, diff_bearing)[:2]
            particles[i].append(diffused)

    # Calculate ensemble mean trajectory
    mean_trajectory = []
    for step_idx in range(DURATION_HOURS + 1):
        step_lats = [particles[p][step_idx][0] for p in range(PARTICLE_COUNT)]
        step_lons = [particles[p][step_idx][1] for p in range(PARTICLE_COUNT)]
        mean_trajectory.append([float(np.mean(step_lons)), float(np.mean(step_lats))]) # [lon, lat]

    # Final origin point (ensemble mean at T-12h)
    final_pred_lon, final_pred_lat = mean_trajectory[-1]

    # Final origin particle cluster points
    final_points = [
        (particles[p][-1][1], particles[p][-1][0]) # (lon, lat)
        for p in range(PARTICLE_COUNT)
    ]
    
    # Convex Hull of final origin particle cloud with a 150m safety buffer
    pts_geom = MultiPoint(final_points)
    hull = pts_geom.convex_hull

    project_to_meters = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    project_to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    
    hull_m = transform(project_to_meters, hull)
    buffered_hull_m = hull_m.buffer(150.0) # 150m buffer in metric projection
    final_origin_polygon = transform(project_to_wgs84, buffered_hull_m)
    
    # Calculate effective uncertainty radius (max geodesic distance from mean to any particle)
    radii = [
        geodesic((final_pred_lat, final_pred_lon), (lat, lon)).meters
        for (lon, lat) in final_points
    ]
    effective_uncertainty_radius_m = float(np.max(radii)) + 150.0

    # Swept Corridor (buffer of ensemble path in metric projection)
    mean_line = LineString(mean_trajectory)
    line_m = transform(project_to_meters, mean_line)
    corridor_m = line_m.buffer(effective_uncertainty_radius_m)
    swept_corridor = transform(project_to_wgs84, corridor_m)

    # Assemble Payload
    payload = {
        "spill_id": "SPILL_TEST3_001",
        "data_mode": "real_observation_with_parameter_driven_simulation",
        
        "slick_observation_time_utc": "2026-09-02T00:00:00Z",
        "estimated_origin_time_utc": "2026-09-01T12:00:00Z",
        "observed_slick_centroid_lat": OBS_LAT,
        "observed_slick_centroid_lon": OBS_LON,

        "predicted_origin_lat": round(final_pred_lat, 6),
        "predicted_origin_lon": round(final_pred_lon, 6),
        "actual_origin_lat": None,
        "actual_origin_lon": None,
        "reference_source": "not_publicly_available",
        "drift_location_error_km": None,
        "evaluation_status": "ground_truth_unavailable",

        # Geospatial Payloads
        "origin_point_geojson": {
            "type": "Point",
            "coordinates": [round(final_pred_lon, 6), round(final_pred_lat, 6)]
        },
        "hindcast_path_geojson": {
            "type": "LineString",
            "coordinates": [[round(x, 6), round(y, 6)] for x, y in mean_trajectory]
        },
        "hindcast_swept_corridor_geojson": mapping(swept_corridor),
        "final_origin_uncertainty_polygon_geojson": mapping(final_origin_polygon),

        # Parameters & Physical Metadata
        "wind_speed_mps": WIND_SPEED_MPS,
        "wind_direction_from_degrees": WIND_DIR_FROM_DEG,
        "current_speed_mps": CURRENT_SPEED_MPS,
        "current_direction_to_degrees": CURRENT_DIR_TO_DEG,
        "wind_drift_coefficient": WIND_DRIFT_COEFF,
        "current_coefficient": CURRENT_COEFF,
        "time_step_seconds": TIMESTEP_SECONDS,
        "duration_hours": DURATION_HOURS,

        "particle_count": PARTICLE_COUNT,
        "random_seed": RANDOM_SEED,
        "uncertainty_method": "monte_carlo_particle_diffusion_convex_hull",
        "horizontal_diffusivity_m2_s": DIFFUSIVITY_M2_S,
        "final_uncertainty_radius_m": round(effective_uncertainty_radius_m, 2),
        "corridor_geometry_meaning": "hindcast_swept_corridor represents the bounding swept area of the ensemble trajectory over 12 hours. final_origin_uncertainty_polygon is the metric buffered convex hull of the 250-particle cluster at T-12h.",
        "geodesic_movement_method": "geopy.distance.geodesic (WGS-84 ellipsoid)",
        "assumptions": [
            "Constant spatial and temporal forcing fields over 12-hour hindcast horizon",
            "Brownian isotropic horizontal turbulent diffusion",
            "Direct geodesic displacements along ellipsoid geodesic paths"
        ]
    }

    os.makedirs("day5_outputs", exist_ok=True)
    out_path = "day5_outputs/sunidhi_real_scenario_final.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nCompleted successfully!")
    print(f"Output saved to: {out_path}")
    print(f"Ensemble Predicted Origin: [{final_pred_lat:.6f}, {final_pred_lon:.6f}]")
    print(f"Effective Particle Spread Radius: {effective_uncertainty_radius_m:.2f} m")

if __name__ == "__main__":
    run_monte_carlo_hindcast()
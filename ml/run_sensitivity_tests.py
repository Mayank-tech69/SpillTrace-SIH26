import json
import math
import os
import time
import numpy as np
from geopy.distance import geodesic

# --- BASE PARAMETERS ---
OBS_LAT = 19.495814
OBS_LON = 71.058396
DURATION_HOURS = 12
TIMESTEP_SECONDS = 3600
PARTICLE_COUNT = 250
DIFFUSIVITY_M2_S = 1.0

def run_simulation(test_name, wind_mps, wind_dir, curr_mps, curr_dir, seed_val):
    start_time = time.time()
    
    np.random.seed(seed_val)
    diffusion_sigma = math.sqrt(2.0 * DIFFUSIVITY_M2_S * TIMESTEP_SECONDS)
    
    # Base advection math
    wind_to = (wind_dir + 180.0) % 360.0
    wind_u = wind_mps * 0.03 * math.sin(math.radians(wind_to))
    wind_v = wind_mps * 0.03 * math.cos(math.radians(wind_to))
    curr_u = curr_mps * 1.0 * math.sin(math.radians(curr_dir))
    curr_v = curr_mps * 1.0 * math.cos(math.radians(curr_dir))
    
    hindcast_u = -(wind_u + curr_u)
    hindcast_v = -(wind_v + curr_v)
    
    step_dist = math.hypot(hindcast_u, hindcast_v) * TIMESTEP_SECONDS
    step_bearing = (math.degrees(math.atan2(hindcast_u, hindcast_v)) + 360.0) % 360.0
    
    # Particles setup
    initial_offsets = np.random.normal(0, 25.0, (PARTICLE_COUNT, 2))
    particles = []
    
    for i in range(PARTICLE_COUNT):
        p_pt = (OBS_LAT, OBS_LON)
        dx, dy = initial_offsets[i]
        init_dist = math.hypot(dx, dy)
        if init_dist > 0:
            init_bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
            p_pt = geodesic(meters=init_dist).destination(p_pt, init_bearing)[:2]
        particles.append(p_pt)
        
    # Hindcast loop
    for _ in range(DURATION_HOURS):
        for i in range(PARTICLE_COUNT):
            current_lat, current_lon = particles[i]
            # Advection
            advected = geodesic(meters=step_dist).destination((current_lat, current_lon), step_bearing)[:2]
            # Diffusion
            diff_x = np.random.normal(0, diffusion_sigma)
            diff_y = np.random.normal(0, diffusion_sigma)
            diff_dist = math.hypot(diff_x, diff_y)
            diff_bearing = (math.degrees(math.atan2(diff_x, diff_y)) + 360.0) % 360.0
            
            particles[i] = geodesic(meters=diff_dist).destination(advected, diff_bearing)[:2]
            
    # Calculate Mean & Radius
    final_lats = [p[0] for p in particles]
    final_lons = [p[1] for p in particles]
    mean_lat = float(np.mean(final_lats))
    mean_lon = float(np.mean(final_lons))
    
    radii = [geodesic((mean_lat, mean_lon), (lat, lon)).meters for (lat, lon) in particles]
    effective_radius = float(np.max(radii)) + 150.0
    
    exec_time = time.time() - start_time
    
    return {
        "test_scenario": test_name,
        "parameters": {
            "wind_speed_mps": wind_mps,
            "wind_direction_from": wind_dir,
            "current_speed_mps": curr_mps,
            "current_direction_to": curr_dir,
            "random_seed": seed_val
        },
        "metrics": {
            "execution_time_seconds": round(exec_time, 4),
            "predicted_origin_lat": round(mean_lat, 6),
            "predicted_origin_lon": round(mean_lon, 6),
            "final_uncertainty_radius_m": round(effective_radius, 2)
        }
    }

def generate_sensitivity_report():
    print("Running Engine Sensitivity and Reproducibility Tests...")
    
    base_wind = 7.7167
    base_curr = 0.6173
    
    results = [
        # Test 1: Base Case
        run_simulation("Baseline Run", base_wind, 270.0, base_curr, 90.0, 42),
        
        # Test 2: Reproducibility (Exact same as Test 1)
        run_simulation("Reproducibility Check (Same Seed)", base_wind, 270.0, base_curr, 90.0, 42),
        
        # Test 3: Sensitivity - High Wind (+20%)
        run_simulation("Sensitivity: High Wind (+20%)", base_wind * 1.2, 270.0, base_curr, 90.0, 42),
        
        # Test 4: Sensitivity - High Current (+20%)
        run_simulation("Sensitivity: High Current (+20%)", base_wind, 270.0, base_curr * 1.2, 90.0, 42),
        
        # Test 5: Seed Variance (Proves stochastic nature)
        run_simulation("Seed Variance (Seed 99)", base_wind, 270.0, base_curr, 90.0, 99)
    ]
    
    os.makedirs("day5_outputs", exist_ok=True)
    out_path = "day5_outputs/sunidhi_sensitivity_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Success! 5 engine validations completed.")
    print(f"Report saved to: {out_path}")

if __name__ == "__main__":
    generate_sensitivity_report()
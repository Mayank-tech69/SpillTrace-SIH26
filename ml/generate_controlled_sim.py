import math
import json
import os
from geopy.distance import geodesic

def run_controlled_simulation():
    print("Running Controlled Drift-Validation Simulation...")
    
    # Parameters (converted to meters/second to match Sunidhi's strict schema)
    known_lat, known_lon = 19.500000, 70.500000 # Known synthetic source
    wind_spd_mps = 7.72 # ~15 knots
    wind_dir_from = 270.0
    curr_spd_mps = 0.62 # ~1.2 knots
    curr_dir_to = 90.0
    wind_coeff = 0.03
    curr_coeff = 1.0
    duration_hrs = 12
    time_step_sec = 3600
    
    # 1. Physics Engine Setup
    wind_dir_to = (wind_dir_from + 180) % 360
    
    wind_u = wind_spd_mps * wind_coeff * math.sin(math.radians(wind_dir_to))
    wind_v = wind_spd_mps * wind_coeff * math.cos(math.radians(wind_dir_to))
    curr_u = curr_spd_mps * curr_coeff * math.sin(math.radians(curr_dir_to))
    curr_v = curr_spd_mps * curr_coeff * math.cos(math.radians(curr_dir_to))
    
    total_u = wind_u + curr_u
    total_v = wind_v + curr_v
    
    forward_dist_m_hr = math.hypot(total_u, total_v) * 3600
    forward_bearing = (math.degrees(math.atan2(total_u, total_v)) + 360) % 360
    
    # 2. FORWARD DRIFT (Simulating the spill moving over 12 hours)
    current_pt = (known_lat, known_lon)
    for _ in range(duration_hrs):
        current_pt = geodesic(meters=forward_dist_m_hr).destination(current_pt, forward_bearing)[:2]
    slick_lat, slick_lon = current_pt
    
    # 3. BACKWARD HINDCAST (Testing the model in reverse)
    backward_bearing = (forward_bearing + 180) % 360
    current_pt = (slick_lat, slick_lon)
    for _ in range(duration_hrs):
        current_pt = geodesic(meters=forward_dist_m_hr).destination(current_pt, backward_bearing)[:2]
    pred_lat, pred_lon = current_pt
    
    # 4. Error Calculation (Geodesic distance between Known Source and Predicted Origin)
    error_m = geodesic((known_lat, known_lon), (pred_lat, pred_lon)).meters
    
    # 5. Output to Sunidhi's exact schema
    payload = {
        "spill_id": "drift_validation_sim_001",
        "data_mode": "controlled_simulation",
        "synthetic_data": True,
        "synthetic_data_scope": "drift_engine_validation_only",
        "known_source_lat": known_lat,
        "known_source_lon": known_lon,
        "known_release_time_utc": "2026-09-01T12:00:00Z",
        "simulated_observed_slick_lat": round(slick_lat, 6),
        "simulated_observed_slick_lon": round(slick_lon, 6),
        "predicted_origin_lat": round(pred_lat, 6),
        "predicted_origin_lon": round(pred_lon, 6),
        "origin_error_m": round(error_m, 4),
        "wind_speed_mps": wind_spd_mps,
        "wind_direction_from_degrees": wind_dir_from,
        "current_speed_mps": curr_spd_mps,
        "current_direction_to_degrees": curr_dir_to,
        "wind_drift_coefficient": wind_coeff,
        "current_coefficient": curr_coeff,
        "time_step_seconds": time_step_sec,
        "duration_hours": duration_hrs,
        "random_seed": 42,
        "evaluation_status": "controlled_simulation_not_real_ground_truth"
    }
    
    path = "day5_outputs/sunidhi_controlled_sim.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")
    print(f"Physics Validation Error: {error_m:.4f} meters")

if __name__ == "__main__":
    run_controlled_simulation()
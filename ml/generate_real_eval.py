import json
import os

def generate_real_scenario():
    print("Generating Real SAR Scenario Evaluation...")
    
    # Load the corridor for the JSON payload
    with open("day5_outputs/origin_corridor.geojson", "r") as f:
        corridor = json.load(f)
        
    payload = {
        "spill_id": "SPILL_001",
        "data_mode": "real_data",
        "predicted_origin_lat": 19.495786,
        "predicted_origin_lon": 70.709083,
        "actual_origin_lat": None,
        "actual_origin_lon": None,
        "reference_source": "not_publicly_available",
        "drift_location_error_km": None,
        "evaluation_status": "ground_truth_unavailable",
        "origin_corridor_geojson": corridor,
        "origin_time_start_utc": "2026-09-01T12:00:00Z", # T-12 hours
        "origin_time_end_utc": "2026-09-02T00:00:00Z",   # T-0 hours
        "drift_mode": "analyst-parameter-driven",
        "assumptions": ["Constant wind and current fields", "Geodesic trajectory approximation"]
    }
    
    path = "day5_outputs/sunidhi_real_scenario.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")

if __name__ == "__main__":
    generate_real_scenario()
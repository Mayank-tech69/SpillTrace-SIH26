import json
import os

def generate_ais_handoff():
    print("Generating backend handoff payload for Pratyush...")
    
    payload = {
        "spill_id": "SPILL_TEST3_001",
        "drift_run_id": "DRIFT_MC250_SEED42_001",
        
        # 1. Explicit Time Window (+/- 2 hours around the 12:00:00Z estimate for AIS filtering)
        "origin_time_window_start_utc": "2026-09-01T10:00:00Z",
        "origin_time_window_end_utc": "2026-09-01T14:00:00Z",
        "estimated_origin_time_utc": "2026-09-01T12:00:00Z",
        
        # 2. Wind/Current Source Metadata (Strictly declared as unavailable/parameter-driven)
        "wind_forcing_source": "unavailable / analyst-parameter-driven",
        "current_forcing_source": "unavailable / analyst-parameter-driven",
        
        # 3. Model Confirmation
        "model_configuration": "250-particle Monte Carlo stochastic simulation with random_seed=42",
        
        # 4 & 5. Artifact Paths
        "final_exported_geometry_path": "day5_outputs/sunidhi_real_scenario_final.json (Extract 'hindcast_swept_corridor_geojson')",
        "final_origin_polygon_path": "day5_outputs/sunidhi_real_scenario_final.json (Extract 'final_origin_uncertainty_polygon_geojson')",
        "final_detector_metadata_path": "day4_outputs/slick_geometry.geojson"
    }
    
    os.makedirs("day5_outputs", exist_ok=True)
    out_path = "day5_outputs/pratyush_ais_handoff.json"
    
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)
        
    print(f"Success! Handoff file saved to: {out_path}")

if __name__ == "__main__":
    generate_ais_handoff()
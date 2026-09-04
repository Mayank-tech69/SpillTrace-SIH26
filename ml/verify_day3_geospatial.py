import json
import os
from shapely.geometry import shape, Point

def verify_and_extract_geometries():
    print("Running Day 3 Geospatial Validation for Sunidhi...\n")
    
    # 1. Load the master JSON payload
    input_path = "day5_outputs/sunidhi_real_scenario_final.json"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run the Monte Carlo script first.")
        return

    with open(input_path, "r") as f:
        data = json.load(f)

    # Extract geometries and coordinates
    path_geo = data["hindcast_path_geojson"]
    poly_geo = data["final_origin_uncertainty_polygon_geojson"]
    pred_lat = data["predicted_origin_lat"]
    pred_lon = data["predicted_origin_lon"]

    # --- SUNIDHI'S 4 VALIDATION CHECKS ---
    
    # Check 1 & 3: GeoJSON Validity & Path Validity (using Shapely)
    path_shape = shape(path_geo)
    poly_shape = shape(poly_geo)
    path_valid = path_shape.is_valid
    poly_valid = poly_shape.is_valid

    # Check 2: Backward Direction Consistency (Wind 270, Current 90 -> Advection MUST be purely West)
    # If moving West, longitude must consistently decrease every hour.
    lons = [pt[0] for pt in path_geo["coordinates"]]
    is_westward_consistent = all(lons[i] >= lons[i+1] for i in range(len(lons)-1))

    # Check 4: Predicted Origin strictly inside the Uncertainty Polygon
    pred_point = Point(pred_lon, pred_lat)
    is_origin_in_poly = pred_point.within(poly_shape)

    # --- PRINT RESULTS ---
    print("--- VALIDATION RESULTS ---")
    print(f"1. Path GeoJSON Validity:           {'PASS' if path_valid else 'FAIL'}")
    print(f"2. Polygon GeoJSON Validity:        {'PASS' if poly_valid else 'FAIL'}")
    print(f"3. Backward Direction Consistent:   {'PASS' if is_westward_consistent else 'FAIL'} (Pure Westward drift)")
    print(f"4. Origin strictly inside Polygon:  {'PASS' if is_origin_in_poly else 'FAIL'}\n")

    # --- EXPORT STANDALONE FILES ---
    path_out = "day5_outputs/SPILL_TEST3_001_hindcast_path.geojson"
    poly_out = "day5_outputs/SPILL_TEST3_001_uncertainty_polygon.geojson"

    with open(path_out, "w") as f:
        json.dump(path_geo, f, indent=4)
        
    with open(poly_out, "w") as f:
        json.dump(poly_geo, f, indent=4)

    print("--- EXPORT COMPLETE ---")
    print(f"Saved: {path_out}")
    print(f"Saved: {poly_out}")

if __name__ == "__main__":
    verify_and_extract_geometries()
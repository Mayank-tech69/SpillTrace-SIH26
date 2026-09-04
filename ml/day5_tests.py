from day5_hindcast import calculate_drift_vector

def run_tests():
    print("Running Day 5 Unit Tests...")
    
    # Test 1: Eastward forces must produce Westward hindcast
    bearing, _ = calculate_drift_vector(wind_spd=10, wind_dir=270, curr_spd=1, curr_dir=90)
    assert 269.9 <= bearing <= 270.1, f"Expected 270.0, got {bearing}"
    print("[PASS] Eastward vectors properly hindcast Westward (270°)")

    # Test 2: Purely East/West forces produce zero North/South movement
    bearing, _ = calculate_drift_vector(wind_spd=0, wind_dir=0, curr_spd=1.5, curr_dir=90)
    assert bearing == 270.0, f"Expected 270.0, got {bearing}"
    print("[PASS] No northward component produces 0° latitude movement")

    # Test 3: Reversing the forces (Northward forces must hindcast Southward)
    bearing, _ = calculate_drift_vector(wind_spd=0, wind_dir=0, curr_spd=1.5, curr_dir=0)
    assert bearing == 180.0, f"Expected 180.0, got {bearing}"
    print("[PASS] Northward vectors properly hindcast Southward (180°)")

    print("\nAll deterministic unit tests passed successfully!")

if __name__ == "__main__":
    run_tests()
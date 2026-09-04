import json
import os

def generate_fastapi_contract():
    print("Generating FastAPI Detector Contract for Aayush...")
    
    # Reading the Day 4 metadata to include in the payload
    try:
        with open("day4_outputs/slick_geometry_metadata.json", "r") as f:
            sample_metadata = json.load(f)
    except FileNotFoundError:
        sample_metadata = {"status": "placeholder_run_day4_pipeline_first"}

    contract = {
        "1_detector_entry_point": {
            "module": "sar_inference.py",
            "function": "process_sar_scene(file_path: str, scene_id: str)",
            "expected_input": "Local file path to downloaded SAR .tiff",
            "input_requirements": "1-channel (VV or VH) or 3-channel (RGB representation) SAR intensity image. Dynamic normalization handled internally.",
            "execution_mode": "Asynchronous (recommended for FastAPI via Celery/Redis due to inference time)",
            "estimated_runtime": "5-15 seconds per scene depending on hardware"
        },
        
        "2_model_metadata": {
            "detector_name": "SpillTrace Segmentation Engine",
            "model_architecture": "UNet",
            "active_checkpoint": "unet_final_spilltrace.keras",
            "class_mapping": {"0": "Background/Sea", "1": "Oil Spill", "2": "Look-alike", "3": "Land/Ship"},
            "oil_spill_class_index": 1,
            "classification_method": "Softmax with configurable threshold",
            "probability_threshold": 0.3,
            "morphology_cleanup": "cv2.MORPH_OPEN and cv2.MORPH_CLOSE applied. Connected components < min_area_threshold removed.",
            "fallback_behavior": "If checkpoint fails/missing, return HTTP 503 with error details."
        },
        
        "3_output_artifacts": {
            "storage_strategy": "Artifacts saved locally to configured output directory. Paths returned in JSON response.",
            "final_oil_mask": "{output_dir}/{scene_id}_oil_mask.png",
            "probability_map": "{output_dir}/{scene_id}_prob_heatmap.png",
            "slick_geojson": "{output_dir}/{scene_id}_slick_geometry.geojson"
        },
        
        "4_geojson_geometry_contract": {
            "crs": "EPSG:4326 (WGS 84)",
            "coordinate_order": "[longitude, latitude]",
            "polygon_validity": "Repaired and verified via Shapely (is_valid == True)",
            "area_metric": "Geodesic area reported in square kilometres (km^2)",
            "perimeter_metric": "Geodesic perimeter reported in metres (m)",
            "multipart_handling": "Multiple slicks represented as a FeatureCollection of multiple Polygon features."
        },
        
        "5_processing_and_error_contract": {
            "job_states": ["QUEUED", "PROCESSING", "COMPLETED", "FAILED"],
            "job_id_format": "UUID-v4",
            "empty_detection_behavior": "If no oil detected, status is COMPLETED. GeoJSON returns empty FeatureCollection. NOT an error.",
            "error_codes": {
                "ERR_INVALID_FILE": "File is corrupted or not a valid GeoTIFF.",
                "ERR_MISSING_CRS": "GeoTIFF lacks spatial referencing.",
                "ERR_MODEL_LOAD": "Keras checkpoint failed to load into memory."
            }
        },
        
        "6_example_payloads": {
            "valid_request": {
                "scene_id": "SAR_20260901_001",
                "file_path": "/storage/inputs/test3.tiff"
            },
            "successful_response": {
                "status": "COMPLETED",
                "scene_id": "SAR_20260901_001",
                "message": "Oil slick detected successfully.",
                "artifacts": {
                    "geojson": "/storage/outputs/SAR_20260901_001_slick_geometry.geojson",
                    "metadata": sample_metadata
                }
            },
            "no_oil_response": {
                "status": "COMPLETED",
                "scene_id": "SAR_20260901_002",
                "message": "Processing successful. No oil pixels detected.",
                "artifacts": {
                    "geojson": None,
                    "metadata": {"total_slicks_detected": 0}
                }
            },
            "failure_response": {
                "status": "FAILED",
                "scene_id": "SAR_20260901_003",
                "error_code": "ERR_MISSING_CRS",
                "message": "GeoTIFF lacks EPSG spatial referencing."
            }
        }
    }

    os.makedirs("day5_outputs", exist_ok=True)
    out_path = "day5_outputs/aayush_fastapi_contract.json"
    with open(out_path, "w") as f:
        json.dump(contract, f, indent=4)
        
    print(f"Handoff contract saved to: {out_path}")

if __name__ == "__main__":
    generate_fastapi_contract()
from flask import Blueprint, jsonify
import csv
import os

data_bp = Blueprint("data", __name__)

# --- Step 1: Create a global variable to hold the data ---
EXOPLANET_DATA = []

# --- Step 2: Define the path and load the data ONCE at startup ---
CSV_FILE = os.path.join(os.path.dirname(__file__), "..", "../server/merged_exoplanet_dataset.csv")

try:
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # --- Step 3: Convert text from CSV into proper numbers (float/int) ---
            try:
                # Convert all potentially numeric fields
                numeric_fields = ['period_days', 'duration_hours', 'depth_ppm', 
                                  'planet_radius_rearth', 'score', 'duty_cycle', 
                                  'transit_frequency', 'planet_radius_from_depth']
                for field in numeric_fields:
                    if row[field]: # Check if the string is not empty
                        row[field] = float(row[field])

                # Convert integer fields
                if row['mission_encoded']:
                    row['mission_encoded'] = int(row['mission_encoded'])

            except (ValueError, KeyError) as e:
                # This handles cases where a row might have bad data or a missing column
                print(f"Skipping row due to data conversion error: {e}, row: {row}")
                continue # Skip to the next row
            
            EXOPLANET_DATA.append(row)

except FileNotFoundError:
    print(f"FATAL ERROR: The data file was not found at {CSV_FILE}")
    # In a real app, you might want to exit or handle this more gracefully
    EXOPLANET_DATA = [{"error": "Data file not found on server"}]


@data_bp.route("/data")
def get_data():
    """
    This function no longer reads the file. It just returns the
    data that was already loaded into the EXOPLANET_DATA list.
    """
    return jsonify(EXOPLANET_DATA)
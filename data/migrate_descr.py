#!/usr/bin/env python3
"""
Migration script to split the central descr.yml into individual descr.yml files
in each database folder.
"""

import os

import yaml

# Read the central descr.yml
with open("descr.yml", "r") as f:
    central_descr = yaml.safe_load(f)

# Process each dataset
for dataset_key, dataset_info in central_descr.items():
    # Extract information from the list-based format
    # Current format: [{name: ...}, {target: ...}, {descr: ...}, {access: ...}, {license: ...}]
    name = dataset_info[0]["name"]
    target = dataset_info[1]["target"]
    descr = dataset_info[2]["descr"]
    access = dataset_info[3]["access"]
    license_info = dataset_info[4].get("license", "unknown")

    # Create the simplified dictionary format
    new_descr = {
        "name": name,
        "target": target,
        "description": descr,
        "access": access,
        "license": license_info,
    }

    # Check if the folder exists
    folder_path = name
    if os.path.isdir(folder_path):
        # Write the new descr.yml file
        descr_file_path = os.path.join(folder_path, "descr.yml")
        with open(descr_file_path, "w") as f:
            yaml.dump(new_descr, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Created {descr_file_path}")
    else:
        print(f"⚠ Warning: Folder '{folder_path}' does not exist for dataset '{name}'")

print("\nMigration complete!")
print("Note: The central descr.yml has been kept for reference.")
print("You can run make_readme.py to test the new individual descr.yml files.")

import json
import argparse
from collections import Counter

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Process two JSONL files and calculate duplicate coordinates between them."
)
parser.add_argument("--j1", help="Path to the first JSONL file")
parser.add_argument("--j2", help="Path to the second JSONL file")
args = parser.parse_args()

def extract_coordinates(jsonl_file):
    """Extract coordinates from a JSONL file."""
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        result = json.load(f)
        for line in result:
            data.append(line[0])  # Parse each JSON line

    # Extract coordinates
    coordinates = []
    for entry in data:
        coordinates.extend([(item[0], item[1]) for item in entry["ig_gold"]])
    return coordinates

# Extract coordinates from both files
coordinates_file1 = extract_coordinates(args.j1)
coordinates_file2 = extract_coordinates(args.j2)

# Count occurrences in both files
counter_file1 = Counter(coordinates_file1)
counter_file2 = Counter(coordinates_file2)

# Find duplicates between the two files
total_coordinates_combined = set(coordinates_file1) | set(coordinates_file2)
duplicate_coordinates = set(coordinates_file1) & set(coordinates_file2)
duplicate_count = len(duplicate_coordinates)

total_coordinates_file1 = len(set(coordinates_file1))
total_coordinates_file2 = len(set(coordinates_file2))
total_coordinates_combined = len(total_coordinates_combined)

# Calculate duplicate ratios
duplicate_ratio_file1 = duplicate_count / total_coordinates_file1 if total_coordinates_file1 > 0 else 0
duplicate_ratio_file2 = duplicate_count / total_coordinates_file2 if total_coordinates_file2 > 0 else 0
duplicate_ratio_combined = duplicate_count / total_coordinates_combined if total_coordinates_combined > 0 else 0

# Print results
print(f"Total coordinates in file 1: {total_coordinates_file1}")
print(f"Total coordinates in file 2: {total_coordinates_file2}")
print(f"Duplicate coordinates between files: {duplicate_count}")
print(f"Ratio of duplicates (file 1): {duplicate_ratio_file1:.2%}")
print(f"Ratio of duplicates (file 2): {duplicate_ratio_file2:.2%}")
print(f"Ratio of duplicates (combined): {duplicate_ratio_combined:.2%}")

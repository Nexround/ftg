import json
import argparse
from collections import Counter

from module.func import scatter_plot, plot_points

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


def filter_counter(counter_obj):
    # 计算平均值
    mean_value = int(sum(counter_obj.values()) / len(counter_obj))
    threshold = int(mean_value * 0)
    # threshold = int(mean_value * 0.1)
    # 过滤元素
    filtered_counter = {
        key: value for key, value in counter_obj.items() if value >= threshold
    }

    return filtered_counter


# Extract coordinates from both files
coordinates_file1 = extract_coordinates(args.j1)
coordinates_file2 = extract_coordinates(args.j2)

# Count occurrences in both files
counter_file1 = Counter(coordinates_file1)
counter_file2 = Counter(coordinates_file2)
# counter_file1 = filter_counter(counter_file1)
counter_file2 = filter_counter(counter_file2)
# scatter_plot(counter_file1)
# scatter_plot(counter_file2)

# for key in list(counter_file1.keys()):
#     if counter_file1[key] == 1:
#         del counter_file1[key]
# for key in list(counter_file2.keys()):
#     if counter_file2[key] == 1:
#         del counter_file2[key]

# Find duplicates between the two files
total_coordinates_combined = set(counter_file1) | set(counter_file2)
duplicate_coordinates = set(counter_file1) & set(counter_file2)
complement_1 = set(counter_file1) - set(counter_file2)
complement_2 = set(counter_file2) - set(counter_file1)
# plot_points(difference_coordinates)

duplicate_count = len(duplicate_coordinates)

total_coordinates_file1 = len(set(counter_file1))
total_coordinates_file2 = len(set(counter_file2))
total_coordinates_combined = len(total_coordinates_combined)
total_complement_1 = len(complement_1)
total_complement_2 = len(complement_2)
# Calculate duplicate ratios
duplicate_ratio_file1 = (
    duplicate_count / total_coordinates_file1 if total_coordinates_file1 > 0 else 0
)
duplicate_ratio_file2 = (
    duplicate_count / total_coordinates_file2 if total_coordinates_file2 > 0 else 0
)
duplicate_ratio_combined = (
    duplicate_count / total_coordinates_combined
    if total_coordinates_combined > 0
    else 0
)
# # Find duplicates between the two files
# total_coordinates_combined = set(coordinates_file1) | set(coordinates_file2)
# duplicate_coordinates = set(coordinates_file1) & set(coordinates_file2)
# duplicate_count = len(duplicate_coordinates)

# total_coordinates_file1 = len(set(coordinates_file1))
# total_coordinates_file2 = len(set(coordinates_file2))
# total_coordinates_combined = len(total_coordinates_combined)

# # Calculate duplicate ratios
# duplicate_ratio_file1 = (
#     duplicate_count / total_coordinates_file1 if total_coordinates_file1 > 0 else 0
# )
# duplicate_ratio_file2 = (
#     duplicate_count / total_coordinates_file2 if total_coordinates_file2 > 0 else 0
# )
# duplicate_ratio_combined = (
#     duplicate_count / total_coordinates_combined
#     if total_coordinates_combined > 0
#     else 0
# )
import json

# 将集合转换为列表，因为 JSON 不支持集合
complement_1_list = list(complement_1)

# 写入 JSON 文件
with open('complement_1.json', 'w', encoding='utf-8') as f:
    json.dump(complement_1_list, f, ensure_ascii=False, indent=4)

print("写入完成！")

# Print results
print(f"Total coordinates in file 1: {total_coordinates_file1}")
print(f"Total coordinates in file 2: {total_coordinates_file2}")
print(f"Duplicate coordinates between files: {duplicate_count}")
print(f"Difference coordinates_1: {total_complement_1}")
print(f"Difference coordinates_2: {total_complement_2}")
print(f"Ratio of duplicates (file 1): {duplicate_ratio_file1:.2%}")
print(f"Ratio of duplicates (file 2): {duplicate_ratio_file2:.2%}")
print(f"Ratio of duplicates (combined): {duplicate_ratio_combined:.2%}")

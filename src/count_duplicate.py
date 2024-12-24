import json
import argparse
from collections import Counter

data = []

parser = argparse.ArgumentParser(
    description="Process a JSONL file and calculate duplicate coordinates."
)
parser.add_argument("jsonl_file", help="Path to the JSONL file")
args = parser.parse_args()

with open(args.jsonl_file, "r", encoding="utf-8") as f:
    result = json.load(f)
    for line in result:
        # cleaned_list = remove_empty_dimensions(line)
        data.append(line[0])  # 逐行解析 JSON

# Extract coordinates
coordinates = []
for entry in data:
    coordinates.extend([(item[0], item[1]) for item in entry["ig_gold"]])
# append() 向列表末尾添加一个元素，可以是任何数据类型（包括列表、字典等）。
# extend() 向列表末尾添加多个元素，通常用于将另一个可迭代对象的元素逐一添加到列表中。
# Count occurrences of each coordinate
coordinate_counts = Counter(coordinates)
# dict_coordinate_counts = [{"pos": pos, "count": count} for pos, count in coordinate_counts.items()]
# with open("coordinate_counts.json", "w", encoding="utf-8") as f:
#     json.dump(dict_coordinate_counts, f, ensure_ascii=False, indent=4)
# Calculate total and duplicates
total_coordinates = len(coordinates)
duplicate_count = sum(1 for _, count in coordinate_counts.items() if count > 1)
duplicate_ratio = duplicate_count / total_coordinates

# Print results
print(f"Total coordinates: {total_coordinates}")
print(f"Duplicate coordinates: {duplicate_count}")
print(f"Ratio of duplicates: {duplicate_ratio:.2%}")

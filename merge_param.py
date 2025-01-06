import json
import argparse
import glob

parser = argparse.ArgumentParser(description='Merge multi params files.')

parser.add_argument('--input_pattern', type=str, required=True, help='input param files, e.g. AtAl_3_3.deepgeco*.params')
parser.add_argument('--output_file', type=str, required=True, help='output param file, e.g. AtAl_3_3.deepgeco.params')

FLAGS = parser.parse_args()

input_files = FLAGS.input_pattern
output_file = FLAGS.output_file
merged_data = {}

json_files = glob.glob(input_files)

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

        for key, value in data.items():
            if key not in merged_data:
                merged_data[key] = value

with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)

print(f"Merge finished, -> '{output_file}'")
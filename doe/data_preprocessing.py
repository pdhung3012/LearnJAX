import json
import random
import os
from lxml import etree

def extract_outage_tags(xml_path):
    # Register the 'po' namespace
    ns = {'po': 'http://iec.ch/TC57/2014/PubOutages#'}

    # Parse the XML
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Find all <po:Outage> elements
    outages = root.findall('.//po:Outage', namespaces=ns)

    # Convert each <po:Outage> element to a string
    outage_strings = [etree.tostring(outage, pretty_print=True, encoding='unicode') for outage in outages]
    return outage_strings


def split_json_data(
    input_path,
    label_path,
    output_dir,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    # Sanity check
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    # Load JSON data
    with open(input_path, 'r') as f:
        data = json.load(f)
    outage_entries = extract_outage_tags(xml_file_path)
    data_2=[]
    for ind in range(0,len(data)):
        new_obj={}
        new_obj['index'] = ind+1
        new_obj['input']=str(data[ind])
        new_obj['output']=outage_entries[ind]
        data_2.append(new_obj)
    data=data_2

    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(data)

    # Compute split indices
    total = len(data)
    train_end = int(train_ratio * total)
    valid_end = train_end + int(valid_ratio * total)

    # Split data
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data  = data[valid_end:]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    with open(os.path.join(output_dir, 'input_train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(output_dir, 'input_valid.json'), 'w') as f:
        json.dump(valid_data, f, indent=2)
    with open(os.path.join(output_dir, 'input_test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"Data split complete. Saved to {output_dir}")

# Example usage:
xml_file_path = '../../pair-pair-dataset/output_standardized_march_pretty.xml'
split_json_data("../../pair-pair-dataset/input_nonstandard_march.json",xml_file_path, '../../preprocess-dataset/')
# Example usage


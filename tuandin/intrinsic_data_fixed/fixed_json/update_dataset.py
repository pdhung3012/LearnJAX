#!/usr/bin/env python3
"""
Script to update LLM_fix_output and Input_Code fields in fix_bugs_dataset.json
from corresponding fixed JAX code files and PyTorch code files.
"""
import json
import os
from pathlib import Path

def update_dataset_from_verification(json_path, verification_dir):
    """
    Update LLM_fix_output and Input_Code fields from verification files.
    
    Updates:
    - LLM_fix_output from verification/{example_id}/jax_code_fixed.py
    - Input_Code from verification/{example_id}/pytorch_code.py
    
    Args:
        json_path: Path to fix_bugs_dataset.json
        verification_dir: Path to verification directory
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    updated_fix_count = 0
    updated_input_count = 0
    missing_fixed_files = []
    missing_pytorch_files = []
    
    # Iterate through examples
    for example in data['Pytorch_to_JAX_Examples']:
        example_id = example['Example_id']
        
        # Path to the fixed JAX code
        fixed_file_path = Path(verification_dir) / example_id / 'jax_code_fixed.py'
        # Path to the PyTorch code
        pytorch_file_path = Path(verification_dir) / example_id / 'pytorch_code.py'
        
        # Update LLM_fix_output from jax_code_fixed.py
        if fixed_file_path.exists():
            with open(fixed_file_path, 'r') as f:
                fixed_code = f.read()
            example['LLM_fix_output'] = fixed_code
            updated_fix_count += 1
            print(f"✓ Updated LLM_fix_output for {example_id}")
        else:
            missing_fixed_files.append(example_id)
            print(f"✗ Missing jax_code_fixed.py: {fixed_file_path}")
        
        # Update Input_Code from pytorch_code.py
        if pytorch_file_path.exists():
            with open(pytorch_file_path, 'r') as f:
                pytorch_code = f.read()
            example['Input_Code'] = pytorch_code
            updated_input_count += 1
            print(f"✓ Updated Input_Code for {example_id}")
        else:
            missing_pytorch_files.append(example_id)
            print(f"✗ Missing pytorch_code.py: {pytorch_file_path}")
    
    # Save the updated JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Updated LLM_fix_output: {updated_fix_count}/{len(data['Pytorch_to_JAX_Examples'])} examples")
    print(f"  Updated Input_Code:     {updated_input_count}/{len(data['Pytorch_to_JAX_Examples'])} examples")
    print(f"  Missing jax_code_fixed.py: {len(missing_fixed_files)} examples")
    print(f"  Missing pytorch_code.py:   {len(missing_pytorch_files)} examples")
    
    if missing_fixed_files:
        print(f"\n  Missing jax_code_fixed.py for: {', '.join(missing_fixed_files)}")
    if missing_pytorch_files:
        print(f"  Missing pytorch_code.py for: {', '.join(missing_pytorch_files)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    # Adjust these paths as needed
    json_path = "/Users/tuandinh/coop_hung/fhudfdkefnewowo/fixed_data/intrinsic/fix_bugs_dataset.json"
    verification_dir = "/Users/tuandinh/coop_hung/fhudfdkefnewowo/all_data/fixed_bug_dataset/verification"
    
    update_dataset_from_verification(json_path, verification_dir)
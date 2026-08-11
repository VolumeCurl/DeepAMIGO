import json
import os

def load_group_indices(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Index file not found: {json_path}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    indices_dict = {}
    

    valid_keys = ['MF', 'BP', 'CC']
    
    for key in valid_keys:
        if key in data:
            indices_dict[key] = [int(i) for i in data[key]]
            print(f"   Category {key}: {len(indices_dict[key])} classes")
            
    return indices_dict
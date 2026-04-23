import h5py
import json

def remove_quantization_mode(d):
    modified = False
    if isinstance(d, dict):
        if 'quantization_mode' in d:
            del d['quantization_mode']
            modified = True
        for k, v in d.items():
            if remove_quantization_mode(v):
                modified = True
    elif isinstance(d, list):
        for item in d:
            if remove_quantization_mode(item):
                modified = True
    return modified

def fix_h5_file(file_path):
    print(f"Opening {file_path}...")
    with h5py.File(file_path, 'r+') as f:
        for attr_name in f.attrs:
            attr_val = f.attrs[attr_name]
            if isinstance(attr_val, bytes):
                try:
                    attr_str = attr_val.decode('utf-8')
                    config = json.loads(attr_str)
                    if remove_quantization_mode(config):
                        print(f"Found and removed 'quantization_mode' in {attr_name}!")
                        new_config_str = json.dumps(config).encode('utf-8')
                        f.attrs[attr_name] = new_config_str
                except:
                    pass
                    
        # Also check model_weights attributes
        if 'model_weights' in f:
            for attr_name in f['model_weights'].attrs:
                attr_val = f['model_weights'].attrs[attr_name]
                if isinstance(attr_val, bytes):
                    try:
                        attr_str = attr_val.decode('utf-8')
                        config = json.loads(attr_str)
                        if remove_quantization_mode(config):
                            print(f"Found and removed 'quantization_mode' in model_weights {attr_name}!")
                            new_config_str = json.dumps(config).encode('utf-8')
                            f['model_weights'].attrs[attr_name] = new_config_str
                    except:
                        pass

if __name__ == "__main__":
    try:
        fix_h5_file("agroware_model.h5")
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")

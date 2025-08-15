import h5py
import os

def print_tree_recursively(group, prefix=""):
    """Recursively prints the HDF5 structure with tree-like connectors."""
    # Get items in the group and convert to a list to know the count
    items = list(group.items())
    num_items = len(items)
    
    for i, (name, obj) in enumerate(items):
        is_last = (i == num_items - 1)
        connector = "└── " if is_last else "├── "
        
        if isinstance(obj, h5py.Group):
            print(f"{prefix}{connector}📂 {name}/")
            # The prefix for children needs to be updated
            child_prefix = prefix + ("    " if is_last else "│   ")
            print_tree_recursively(obj, child_prefix)
        elif isinstance(obj, h5py.Dataset):
            print(f"{prefix}{connector}📄 {name} (Shape: {obj.shape}, Dtype: {obj.dtype})")

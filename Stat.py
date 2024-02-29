import h5py

def print_hdf5_structure(group, indent=0):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print("  " * indent + f"Group: {key}/")
            print_hdf5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print("  " * indent + f"Dataset: {key} - Shape: {item.shape}, Dtype: {item.dtype}")

def print_hdf5_statistics(group):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print_hdf5_statistics(item)
        elif isinstance(item, h5py.Dataset):
            print(f"Dataset: {key}")
            print(f"  Shape: {item.shape}")
            print(f"  Dtype: {item.dtype}")
            print()

def main():
    hdf5_file_path = "test_30400.hdf5"

    try:
        with h5py.File(hdf5_file_path, 'r') as hdf5_file:
            print("HDF5 Structure:")
            print_hdf5_structure(hdf5_file)
            print("\nHDF5 Statistics:")
            print_hdf5_statistics(hdf5_file)
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")

if __name__ == "__main__":
    main()

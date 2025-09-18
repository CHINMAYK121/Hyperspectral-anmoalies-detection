import h5py
import scipy.io
import numpy as np
import os
from pathlib import Path
import re

def explore_hdf5_structure(file_path, max_depth=3, current_depth=0):
    """
    Explore and print the structure of HDF5 file to understand data organization
    """
    def print_structure(name, obj, depth=0):
        indent = "  " * depth
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}Dataset: {name} - Shape: {obj.shape}, Type: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}Group: {name}")
            if depth < max_depth:
                for key in obj.keys():
                    print_structure(f"{name}/{key}", obj[key], depth + 1)
    
    with h5py.File(file_path, 'r') as f:
        print(f"Structure of {file_path}:")
        print_structure('/', f, current_depth)

def convert_he5_to_mat(input_file, output_file=None, dataset_path=None, 
                       include_metadata=True, compression=True):
    """
    Convert HDF5/HE5 hyperspectral image to MATLAB format
    
    Parameters:
    -----------
    input_file : str
        Path to input .he5 or .h5 file
    output_file : str, optional
        Path to output .mat file. If None, uses input filename with .mat extension
    dataset_path : str, optional
        Specific path to hyperspectral data in HDF5 file. If None, attempts to find automatically
    include_metadata : bool
        Whether to include metadata/attributes in output file
    compression : bool
        Whether to compress the output .mat file
    """
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('.mat'))
    
    print(f"Converting {input_file} to {output_file}")
    
    try:
        with h5py.File(input_file, 'r') as hdf_file:
            data_dict = {}

            # Helper: sanitize MATLAB variable names (<=31 chars, start with letter, unique)
            used_names = set()
            def sanitize_mat_key(raw_name: str) -> str:
                # Replace separators and invalid chars
                name = raw_name.replace('/', '_').replace(' ', '_').replace('-', '_').strip('_')
                name = re.sub(r"[^A-Za-z0-9_]", "_", name)
                # Must start with a letter
                if not name or not name[0].isalpha():
                    name = f"v_{name}" if name else "v"
                # Trim to 31 chars
                base = name[:31]
                name = base
                # Ensure uniqueness by appending _1, _2 ... within 31 chars
                if name in used_names:
                    idx = 1
                    while True:
                        suffix = f"_{idx}"
                        candidate = (base[: 31 - len(suffix)]) + suffix
                        if candidate not in used_names:
                            name = candidate
                            break
                        idx += 1
                used_names.add(name)
                return name
            
            # Function to recursively extract data
            def extract_data(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Convert dataset name to valid MATLAB variable name
                    var_name = sanitize_mat_key(name)
                    
                    # Read the data
                    data = obj[:]
                    
                    # Handle different data types
                    if data.dtype.kind in ['S', 'U']:  # String data
                        try:
                            data = data.astype(str)
                        except:
                            data = str(data)
                    
                    data_dict[var_name] = data
                    print(f"  Extracted dataset: {var_name} - Shape: {data.shape}")
                    
                    # Extract attributes if requested
                    if include_metadata and obj.attrs:
                        attrs_dict = {}
                        used_attr_names = set()
                        for attr_name, attr_value in obj.attrs.items():
                            try:
                                if isinstance(attr_value, (bytes, np.bytes_)):
                                    attr_value = attr_value.decode('utf-8')
                                elif isinstance(attr_value, np.ndarray):
                                    attr_value = attr_value.tolist()
                            except:
                                attr_value = str(attr_value)
                            # Sanitize attribute field name for MATLAB struct
                            attr_key = sanitize_mat_key(str(attr_name))
                            # Ensure uniqueness within this struct
                            base = attr_key
                            if attr_key in used_attr_names:
                                idx = 1
                                while True:
                                    suffix = f"_{idx}"
                                    candidate = (base[: 31 - len(suffix)]) + suffix
                                    if candidate not in used_attr_names:
                                        attr_key = candidate
                                        break
                                    idx += 1
                            used_attr_names.add(attr_key)
                            attrs_dict[attr_key] = attr_value
                        
                        if attrs_dict:
                            attr_key = sanitize_mat_key(f"{var_name}_attrs")
                            data_dict[attr_key] = attrs_dict
            
            # If specific dataset path is provided
            if dataset_path:
                if dataset_path in hdf_file:
                    extract_data(dataset_path, hdf_file[dataset_path])
                else:
                    print(f"Warning: Dataset path '{dataset_path}' not found in file")
                    print("Available datasets:")
                    hdf_file.visititems(lambda name, obj: print(f"  {name}") 
                                      if isinstance(obj, h5py.Dataset) else None)
                    return False
            else:
                # Extract all datasets
                hdf_file.visititems(extract_data)
            
            # Extract global attributes
            if include_metadata and hdf_file.attrs:
                global_attrs = {}
                used_global_names = set()
                for attr_name, attr_value in hdf_file.attrs.items():
                    try:
                        if isinstance(attr_value, (bytes, np.bytes_)):
                            attr_value = attr_value.decode('utf-8')
                        elif isinstance(attr_value, np.ndarray):
                            attr_value = attr_value.tolist()
                    except:
                        attr_value = str(attr_value)
                    gkey = sanitize_mat_key(str(attr_name))
                    base = gkey
                    if gkey in used_global_names:
                        idx = 1
                        while True:
                            suffix = f"_{idx}"
                            candidate = (base[: 31 - len(suffix)]) + suffix
                            if candidate not in used_global_names:
                                gkey = candidate
                                break
                            idx += 1
                    used_global_names.add(gkey)
                    global_attrs[gkey] = attr_value
                
                if global_attrs:
                    data_dict['global_attributes'] = global_attrs
        
        # Save to MATLAB format
        if data_dict:
            scipy.io.savemat(output_file, data_dict, 
                           do_compression=compression,
                           format='5',  # MATLAB v5 format for better compatibility
                           oned_as='row')  # Save 1D arrays as row vectors
            
            print(f"Successfully converted to {output_file}")
            print(f"Variables saved: {list(data_dict.keys())}")
            return True
        else:
            print("No data found to convert")
            return False
            
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

def find_multiple_datasets(input_file, min_dimensions=2, min_size=1000):
    """
    Find multiple datasets in HDF5 file that could be hyperspectral images
    
    Parameters:
    -----------
    input_file : str
        Path to input .he5 file
    min_dimensions : int
        Minimum number of dimensions for a dataset to be considered
    min_size : int
        Minimum total size for a dataset to be considered
    
    Returns:
    --------
    list : List of dataset paths that could be hyperspectral images
    """
    datasets = []
    
    def find_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Check if dataset could be a hyperspectral image
            if (len(obj.shape) >= min_dimensions and 
                obj.size >= min_size and
                obj.dtype.kind in ['f', 'i', 'u']):  # float, int, or uint
                datasets.append({
                    'path': name,
                    'shape': obj.shape,
                    'dtype': obj.dtype,
                    'size': obj.size
                })
    
    try:
        with h5py.File(input_file, 'r') as hdf_file:
            hdf_file.visititems(find_datasets)
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    # Sort by size (largest first) to prioritize main data arrays
    datasets.sort(key=lambda x: x['size'], reverse=True)
    return datasets

def convert_multiple_datasets_to_numbered_mats(input_file, output_dir=None, 
                                             min_dimensions=2, min_size=1000,
                                             include_metadata=True):
    """
    Check for multiple datasets in .he5 file and save each as numbered .mat files
    
    Parameters:
    -----------
    input_file : str
        Path to input .he5 file
    output_dir : str, optional
        Directory to save numbered .mat files. If None, uses same directory as input file
    min_dimensions : int
        Minimum dimensions for considering a dataset as hyperspectral data
    min_size : int
        Minimum size for considering a dataset
    include_metadata : bool
        Whether to include metadata in output files
    
    Returns:
    --------
    list : List of created .mat file paths
    """
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(input_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all potential datasets
    datasets = find_multiple_datasets(input_file, min_dimensions, min_size)
    
    if not datasets:
        print(f"No suitable datasets found in {input_file}")
        return []
    
    print(f"Found {len(datasets)} potential hyperspectral datasets in {input_file}")
    
    created_files = []
    
    try:
        with h5py.File(input_file, 'r') as hdf_file:
            
            # Extract global attributes once
            global_attrs = {}
            if include_metadata and hdf_file.attrs:
                for attr_name, attr_value in hdf_file.attrs.items():
                    try:
                        if isinstance(attr_value, (bytes, np.bytes_)):
                            attr_value = attr_value.decode('utf-8')
                        elif isinstance(attr_value, np.ndarray):
                            attr_value = attr_value.tolist()
                        global_attrs[attr_name] = attr_value
                    except:
                        global_attrs[attr_name] = str(attr_value)
            
            for i, dataset_info in enumerate(datasets, 1):
                dataset_path = dataset_info['path']
                output_file = output_dir / f"{i}.mat"
                
                print(f"\nProcessing dataset {i}: {dataset_path}")
                print(f"  Shape: {dataset_info['shape']}, Type: {dataset_info['dtype']}")
                
                # Create data dictionary for this dataset
                data_dict = {}
                
                # Read the main dataset
                dataset = hdf_file[dataset_path]
                data = dataset[:]
                
                # Create meaningful variable names
                var_name = f"data_{i}"
                if 'cube' in dataset_path.lower() or 'image' in dataset_path.lower():
                    var_name = f"hyperspectral_cube_{i}"
                elif 'radiance' in dataset_path.lower():
                    var_name = f"radiance_{i}"
                elif 'reflectance' in dataset_path.lower():
                    var_name = f"reflectance_{i}"

                # Enforce MATLAB key length restriction
                safe_name = var_name[:31]
                data_dict[safe_name] = data
                
                # Add dataset-specific metadata
                if include_metadata and dataset.attrs:
                    attrs_dict = {}
                    for attr_name, attr_value in dataset.attrs.items():
                        try:
                            if isinstance(attr_value, (bytes, np.bytes_)):
                                attr_value = attr_value.decode('utf-8')
                            elif isinstance(attr_value, np.ndarray):
                                attr_value = attr_value.tolist()
                            attrs_dict[attr_name] = attr_value
                        except:
                            attrs_dict[attr_name] = str(attr_value)
                    
                    if attrs_dict:
                        attr_key = (f"{safe_name}_attrs")[:31]
                        data_dict[attr_key] = attrs_dict
                
                # Add file info
                data_dict['file_info'] = {
                    'source_file': os.path.basename(input_file),
                    'dataset_path': dataset_path,
                    'dataset_number': i,
                    'shape': dataset_info['shape'],
                    'dtype': str(dataset_info['dtype']),
                    'total_datasets_in_file': len(datasets)
                }
                
                # Add global attributes
                if global_attrs:
                    data_dict['global_attributes'] = global_attrs
                
                # Try to find associated wavelength data
                wavelength_paths = [
                    dataset_path.replace('Cube', 'Wavelengths').replace('Data', 'Wavelengths'),
                    '/HDFEOS/SWATHS/Hyperion/Data_Fields/Wavelengths',
                    '/Wavelengths',
                    '/Wavelength', 
                    '/Lambda'
                ]
                
                for wl_path in wavelength_paths:
                    if wl_path in hdf_file and wl_path != dataset_path:
                        try:
                            wavelengths = hdf_file[wl_path][:]
                            data_dict['wavelengths'] = wavelengths
                            print(f"  Found associated wavelengths: {wavelengths.shape}")
                            break
                        except:
                            continue
                
                # Save to MATLAB format
                scipy.io.savemat(str(output_file), data_dict,
                               do_compression=True,
                               format='5',
                               oned_as='row')
                
                print(f"  Saved to: {output_file}")
                created_files.append(str(output_file))
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return created_files
    
    print(f"\nSuccessfully created {len(created_files)} .mat files:")
    for file in created_files:
        print(f"  {file}")
    
    return created_files

def convert_hyperspectral_cube(input_file, output_file=None, 
                             cube_dataset=None, wavelengths_dataset=None):
    """
    Specialized function for hyperspectral image cubes
    
    Parameters:
    -----------
    input_file : str
        Path to input .he5 file
    output_file : str, optional
        Output .mat file path
    cube_dataset : str, optional
        Path to hyperspectral cube data (e.g., '/HDFEOS/SWATHS/Hyperion/Data_Fields/Cube')
    wavelengths_dataset : str, optional
        Path to wavelength data (e.g., '/HDFEOS/SWATHS/Hyperion/Data_Fields/Wavelengths')
    """
    
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('.mat'))
    
    try:
        with h5py.File(input_file, 'r') as hdf_file:
            data_dict = {}
            
            # Try to find hyperspectral cube automatically if not specified
            if cube_dataset is None:
                # Common paths for hyperspectral data
                possible_paths = [
                    '/HDFEOS/SWATHS/Hyperion/Data_Fields/Cube',
                    '/Cube',
                    '/Data',
                    '/Image',
                    '/Radiance'
                ]
                
                for path in possible_paths:
                    if path in hdf_file:
                        cube_dataset = path
                        break
                
                if cube_dataset is None:
                    print("Could not automatically find hyperspectral cube. Available datasets:")
                    hdf_file.visititems(lambda name, obj: print(f"  {name}") 
                                      if isinstance(obj, h5py.Dataset) else None)
                    return False
            
            # Read hyperspectral cube
            if cube_dataset in hdf_file:
                cube_data = hdf_file[cube_dataset][:]
                data_dict['hyperspectral_cube'] = cube_data
                print(f"Hyperspectral cube shape: {cube_data.shape}")
                
                # Read wavelengths if available
                if wavelengths_dataset and wavelengths_dataset in hdf_file:
                    wavelengths = hdf_file[wavelengths_dataset][:]
                    data_dict['wavelengths'] = wavelengths
                    print(f"Wavelengths shape: {wavelengths.shape}")
                
                # Try to find wavelengths automatically
                elif wavelengths_dataset is None:
                    wavelength_paths = [
                        '/HDFEOS/SWATHS/Hyperion/Data_Fields/Wavelengths',
                        '/Wavelengths',
                        '/Wavelength',
                        '/Lambda'
                    ]
                    
                    for path in wavelength_paths:
                        if path in hdf_file:
                            wavelengths = hdf_file[path][:]
                            data_dict['wavelengths'] = wavelengths
                            print(f"Found wavelengths at {path}, shape: {wavelengths.shape}")
                            break
                
                # Add basic metadata
                data_dict['file_info'] = {
                    'source_file': os.path.basename(input_file),
                    'cube_shape': cube_data.shape,
                    'data_type': str(cube_data.dtype)
                }
                
                # Save to MATLAB format
                scipy.io.savemat(output_file, data_dict, 
                               do_compression=True,
                               format='5',
                               oned_as='row')
                
                print(f"Successfully saved hyperspectral data to {output_file}")
                return True
            else:
                print(f"Dataset '{cube_dataset}' not found in file")
                return False
                
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

# Example usage functions
def batch_convert(input_directory, output_directory=None, file_pattern="*.he5"):
    """
    Convert multiple HE5 files in a directory
    """
    input_path = Path(input_directory)
    
    if output_directory is None:
        output_path = input_path
    else:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
    
    he5_files = list(input_path.glob(file_pattern))
    
    if not he5_files:
        print(f"No files found matching pattern '{file_pattern}' in {input_directory}")
        return
    
    print(f"Found {len(he5_files)} files to convert")
    
    for he5_file in he5_files:
        output_file = output_path / he5_file.with_suffix('.mat').name
        print(f"\nProcessing: {he5_file.name}")
        convert_he5_to_mat(str(he5_file), str(output_file))

def batch_convert_multiple_datasets(input_directory, output_directory=None, file_pattern="*.he5"):
    """
    Convert multiple HE5 files, each potentially containing multiple datasets
    Creates numbered .mat files for each dataset found
    """
    input_path = Path(input_directory)
    
    if output_directory is None:
        output_path = input_path
    else:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
    
    he5_files = list(input_path.glob(file_pattern))
    
    if not he5_files:
        print(f"No files found matching pattern '{file_pattern}' in {input_directory}")
        return
    
    print(f"Found {len(he5_files)} files to process")
    all_created_files = []
    
    for he5_file in he5_files:
        # Create subdirectory for each input file
        file_output_dir = output_path / he5_file.stem
        file_output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Processing: {he5_file.name}")
        print(f"Output directory: {file_output_dir}")
        
        created_files = convert_multiple_datasets_to_numbered_mats(
            str(he5_file), 
            str(file_output_dir)
        )
        all_created_files.extend(created_files)
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: Created {len(all_created_files)} total .mat files from {len(he5_files)} .he5 files")
    return all_created_files

# Main execution example
if __name__ == "__main__":
    # Example usage:
    
    input_file = "/Users/chinmayk/Desktop/Python day project/PRS_L2D_STD_20201214060713_20201214060717_0001.he5"  # Replace with your file path
    
    if os.path.exists(input_file):
        print("=== Exploring file structure ===")
        explore_hdf5_structure(input_file)
        
        print("\n=== Checking for multiple datasets ===")
        datasets = find_multiple_datasets(input_file)
        
        if len(datasets) > 1:
            print(f"Found {len(datasets)} datasets. Converting to numbered .mat files...")
            convert_multiple_datasets_to_numbered_mats(input_file)
        else:
            print("Found single dataset. Converting to single .mat file...")
            convert_he5_to_mat(input_file)
            
        # Alternative: Batch process multiple files with multiple datasets each
        # batch_convert_multiple_datasets("path/to/input/directory", "path/to/output/directory")
        
    else:
        print(f"File {input_file} not found. Please update the file path.")
        print("\nTo use this script:")
        print("1. Install required packages: pip install h5py scipy numpy")
        print("2. Update the input_file variable with your .he5 file path")
        print("3. Run the script to automatically detect and convert multiple datasets")
        print("\nKey functions:")
        print("- find_multiple_datasets(): Detect multiple hyperspectral datasets")
        print("- convert_multiple_datasets_to_numbered_mats(): Convert to 1.mat, 2.mat, etc.")
        print("- batch_convert_multiple_datasets(): Process multiple .he5 files at once")

# The following example calls were executed unconditionally and caused errors.
# They are intentionally removed to avoid accidental execution on import.
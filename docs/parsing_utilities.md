# Parsing Utilities

This module provides functions to parse, clean, and preprocess annotation files for primate vocalization data.

## Functions Overview

### File Discovery
- `get_annotation_files(directory, extension='.txt')` - Get sorted list of annotation files
- `get_recording_files(directory, extension='.wav')` - Get sorted list of recording files

### Data Loading and Cleaning
- `load_annotation_file(file_path)` - Load single annotation file with standardized columns
- `clean_annotation_dataframe(df)` - Clean and preprocess DataFrame
- `normalize_labels(df, corrections=None)` - Normalize species and call_type labels
- `filter_invalid_entries(df, invalid_species=None, invalid_call_types=None)` - Filter out invalid data

### Analysis and Filtering
- `get_most_common_call_types(df, n=3)` - Get n most common call types
- `filter_by_call_types(df, call_types)` - Filter by specific call types
- `plot_call_type_distribution(df, save_path=None)` - Plot call type distribution

### Complete Pipeline
- `load_and_process_annotations(directory, ...)` - Complete processing pipeline
- `print_dataset_summary(df)` - Print dataset summary statistics
- `save_processed_dataset(df, output_path)` - Save processed data to CSV

## Usage Examples

### Basic Usage
```python
from banana_net.utils.parsing import load_and_process_annotations

# Process all annotations in a directory
dataset = load_and_process_annotations(
    directory='/path/to/annotations',
    filter_top_n=3  # Keep only top 3 call types
)
```

### Custom Processing
```python
# Custom corrections for species-specific issues
corrections = {
    'contact call': 'cc',
    'alarm call': 'ac',
}

# Custom filtering
invalid_species = ['unknown', 'unclear']
invalid_call_types = ['noise', 'background']

dataset = load_and_process_annotations(
    directory='/path/to/annotations',
    corrections=corrections,
    invalid_species=invalid_species,
    invalid_call_types=invalid_call_types,
    filter_top_n=5
)
```

### Step-by-step Processing
```python
from banana_net.utils.parsing import *

# Get files
annotation_files = get_annotation_files('/path/to/data')

# Process each file individually
datasets = []
for file in annotation_files:
    df = load_annotation_file(file)
    df = clean_annotation_dataframe(df)
    datasets.append(df)

# Combine and process
combined_df = pd.concat(datasets, ignore_index=True)
combined_df = normalize_labels(combined_df, corrections)
combined_df = filter_invalid_entries(combined_df, invalid_species, invalid_call_types)

# Analysis
print_dataset_summary(combined_df)
plot_call_type_distribution(combined_df, save_path='distribution.png')
```

## Expected Annotation File Format

The functions expect tab-separated annotation files with these columns:
- `Begin Time (s)` - Start time of vocalization
- `End Time (s)` - End time of vocalization  
- `Low Freq (Hz)` - Lower frequency bound
- `High Freq (Hz)` - Upper frequency bound
- `Inband Power (dB FS)` - Power measurement
- `Species` - Species identifier
- `Call type` - Type of vocalization

Output DataFrame will have standardized column names:
- `begin_time`, `end_time`, `low_freq`, `high_freq`, `inband_power`, `species`, `call_type`, `recording_file`

## Example Script

See `examples/process_species_data.py` for a complete example of batch processing multiple species with different configurations.

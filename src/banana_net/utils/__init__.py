"""
Utilidades para procesamiento y análisis de datos de vocalizaciones de primates.
"""

# Importar funciones principales de cada módulo
from .data_preprocessing import (
    preprocess_annotations, 
    extract_dimensions_by_group, 
    find_potential_typos_per_species,
    correct_species_typos,
    filter_uncommon_combinations,
    sort_and_reset_dataset
)

from .anchor_box_utils import (
    iou,
    calculate_anchors_for_group,
    run_anchor_box_pipeline
)

from .data_visualization import (
    visualize_anchor_results,
    plot_spectrogram_with_boxes,
    plot_distribution_by_group,
    plot_spectral_features
)

from .data_balancing import (
    balance_dataset_by_downsampling,
    split_dataset,
    balance_by_augmentation
)

from .data_augmentation import (
    apply_data_augmentation,
    time_shift,
    frequency_mask,
    time_mask,
    spec_augment,
    apply_augmentation_pipeline
)

# Mantener las versiones antiguas para retrocompatibilidad
from .preprocessing import (
    clean_annotation_dataframe,
    normalize_labels,
    create_feature_set
)

from .loading import (
    get_annotation_files,
    get_recording_files,
    load_single_annotation_file,
    load_annotations_from_species_dirs
)

__all__ = [
    # Data preprocessing
    'preprocess_annotations', 'extract_dimensions_by_group',
    'find_potential_typos_per_species', 'correct_species_typos',
    'filter_uncommon_combinations', 'sort_and_reset_dataset',
    
    # Anchor box utils
    'iou', 'calculate_anchors_for_group', 'run_anchor_box_pipeline',
    
    # Data visualization
    'visualize_anchor_results', 'plot_spectrogram_with_boxes',
    'plot_distribution_by_group', 'plot_spectral_features',
    
    # Data balancing
    'balance_dataset_by_downsampling', 'split_dataset', 'balance_by_augmentation',
    
    # Data augmentation
    'apply_data_augmentation', 'time_shift', 'frequency_mask',
    'time_mask', 'spec_augment', 'apply_augmentation_pipeline',
    
    # Legacy functions
    'clean_annotation_dataframe', 'normalize_labels', 'create_feature_set',
    'get_annotation_files', 'get_recording_files', 'load_single_annotation_file',
    'load_annotations_from_species_dirs'
]
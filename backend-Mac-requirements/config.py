from pathlib import Path

class Config(object):
    GIRDER_MOUNT_FOLDER = Path('girder_mount')
    STUDIES_FOLDER = GIRDER_MOUNT_FOLDER/'collection'
    TEMP_FOLDER = Path("temp")

    # All paths are under the study relative
    # e.g. STUDIES_FOLDER/{study_name}/DATASET_FOLDER
    DATASET_FOLDER = Path('datasets')
    TRAINING_FOLDER = DATASET_FOLDER/'train'
    TESTING_FOLDER = DATASET_FOLDER/'test'    
    

    # Path to save derived data
    APPDATA_FOLDER = Path('derived_data')
    MODELS_FOLDER = Path('models')
    FEATURE_DATA_FOLDER = Path('feature_data')
    ATTENTION_RESULTS_FOLDER = Path('attention_results')
    PATCHES_FOLDER = Path('patches')
    STITCHES_FOLDER = Path('stitches')

    
    ATTENTION_ANNOTATION_FOLDER = ATTENTION_RESULTS_FOLDER/'annotations'
    ATTENTION_HEATMAP_FOLDER = ATTENTION_RESULTS_FOLDER/'heatmaps'
    MODEL_SUFFIX = '.pth'

    FEATURE_EXTRACTOR_NAME = "feature_extractor.pth"
    CLASSIFIER_NAME = "classifier.pth"


    FEATURE_EXTRACTORS = {
        "Resnet-50": "resnet50.a1_in1k",
        "Inception V3": "inception_v3.tf_in1k",
        "MobileNet": "mobilenetv3_large_100.ra_in1k",
        "EfficientNet": "efficientnet_b3.ra2_in1k",
        "Hibou-b": "hibou_b",
    }


from create_patches_fp import create_patches
from extract_features_fp import extract_features
import os

## TODO: make this work with both classes as civ does
def main():
    print("Creating Patches")


    svs_dir = "/Users/parth/datasets/camelyon16/training/normal"
    patch_save_dir = "/Users/parth/datasets/camelyon16/clam/training_normal_patches"
    csv_save_dir = "/Users/parth/datasets/camelyon16/clam/training_normal_patches"
    features_dir = "/Users/parth/datasets/camelyon16/clam/training_normal_features"
    

    classes = [class_name for class_name in os.listdir(svs_dir) if os.path.isdir(os.path.join(svs_dir, class_name))]
    for class_name in classes:
        target_dir = os.path.join(svs_dir, class_name)
        create_patches(source=target_dir, 
                patch_save_dir=patch_save_dir, 
                csv_save_dir=csv_save_dir, 
                patch_size = 256, step_size = 256, patch_level=0, auto_skip=False)
        print("Extracting features")
        extract_features(csv_path=os.path.join(csv_save_dir, "process_list_autogen.csv"),
                        output_dir=features_dir,
                        model_name="resnet50", target_patch_size=224,
                        patches_dir=patch_save_dir, 
                        auto_skip=False, batch_size=256)
    


if __name__ == "__main__":
    main()
import json
import sys
sys.path.append('./backend')
import shutil
import zipfile_deflate64 as zipfile
from flask import Flask, render_template, request, redirect, send_from_directory, url_for, jsonify, send_file, session, Response
from flask_socketio import SocketIO, emit
import torch
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import milui_helper as helper
import openslide
import time
from config import Config
from tqdm import tqdm

from girder_client import GirderClient
from girder_utils.mount_girder import GirderMounter
from girder_utils.file_utils import  copy_folder_girder, create_folder_girder, import_data_to_assetstore_girder, upload_folder_to_girder

## Training packages
from misc_utils import get_wsi_extensions
from preprocessing.patcher import patch
from preprocessing.extractor import Extractor
from pooler import TransMILPooler

# Girder-client
girder_mounter = GirderMounter()
gc  = GirderClient(apiUrl='http://dsa-girder:8080/api/v1')

## TODO: Move this inside `mount` method. The method will be called with every login,
##          and it will authenticate while mounting with new token everytime
gc.authenticate(username="admin",password="password")


# app = Flask(__name__)
app = Flask(__name__, static_folder='../frontend/build', template_folder='../frontend/build')

## TODO: 1. Create a .env file and use secret key from there. 
## TODO: 2. Somehow make it ready for prod
## TODO: 3. Add type hints and default values to all the relevant input boxes in html
app.secret_key = 'tHisisARandomSecretKeyMustChangethissomeTimes'     
app.config.from_object(Config())
socketio = SocketIO(app, cors_allowed_origins="*")

# Initial setup
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024 * 1024  # 250GB limit per chunk (adjust as needed) ## It maybe is limit for a whole file, yet to test
ALLOWED_EXTENSIONS = {'zip'}



# =============================================================================
# Utility/Helper Functions
# =============================================================================

## TODO: Remove this, replaced with girder
def get_datasets_list(): #Unused
    """
    Retrieve the list of available datasets.

    Returns:
        list: A list of dataset names.
    """
    p = Path(Config.DATASET_FOLDER)
    return [x.stem for x in p.iterdir() if x.is_dir()]

def get_models_list():
    """
    Retrieve the list of available models.

    Returns:
        list: A list of model names.
    """
    p = Path(Config.STUDIES_FOLDER)
    models = []
    for dir_name in p.iterdir():
        if dir_name.is_dir:
            classifier = Path(dir_name, "models", 'classifier.pth')
            feature_extractor = Path(dir_name, 'feature_extractor.pth')
            if classifier.exists(): # and feature_extractor.exists():
                models.append(dir_name.stem)
    
    return models

def allowed_file(filename):
    """
    Check if a file has an allowed extension.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    ALLOWED_EXTENSIONS = {'zip'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/v1/get_session_variables/all', methods=['GET', 'POST'])
def get_session_variables_all():
    # print(f'{session=}')
    dict_session = dict(session)
    # print(f'{dict_session=}')
    return jsonify(dict_session)

@app.route('/api/v1/get_session_variables', methods=['POST'])
def get_session_variables():
    session_vars = request.json.get('session_vars', [])

    if not isinstance(session_vars, list):
        return jsonify({'error': 'session_vars must be a list'}), 400

    result = {}
    for var in session_vars:
        if var in session:
            result[var] = session[var]
        else:
            result[var] = None

    return jsonify(result)


@app.route('/api/v1/set_session_variables', methods=['POST'])
def set_session_variables():
    data = request.json
    for key, value in data.items():
        session[key] = value
        print(f"Setting variable `{key}` with value: {value}")

    study_id = None
    if 'study_id' in data:
        study_id = data['study_id']
    elif 'study_id' in session:
        study_id = session['study_id']
    session.modified = True

    # print(f'{dict(data)=}')
    # print(f'{dict(session)=}')
    # print(f'{dir(session)=}')
    # print('----')

    # Save session variable for possiable restoration later
    if study_id:
        gc.addMetadataToCollection(study_id, {'session_variables': dict(session)})
    
    return jsonify({'message': 'session variables set'})

@app.route('/api/v1/set_session_variables/<collection_id>', methods=['POST'])
def set_session_variables_from_saved(collection_id):
    # Get collection
    collection = gc.getCollection(collection_id)
    session_variables = collection['meta']['session_variables']
    print(session_variables)

    # Load all relevent details into session variables
    session.clear()
    for item in session_variables.items():
        session[item[0]] = item[1]
    return Response(status=204)

@app.route('/api/v1/get_available_feature_extractors', methods=['GET'])
def get_available_feature_extractors():
    print(f"returning list of feature extractors: {list(Config.FEATURE_EXTRACTORS.keys())}")
    return jsonify(Config.FEATURE_EXTRACTORS)



# =============================================================================
# Girder Routes/Functions
# =============================================================================

@app.route('/api/v1/mount_client', methods=["POST"])
def mount_girder_client():
    # Making sure to unmount before mounting
    unmount_girder_client()
    print("Received mount request!!\n mounting girder client...")
    girder_token = request.headers.get('Girder-Token')
    
    os.makedirs(Config.GIRDER_MOUNT_FOLDER, exist_ok=True)
    print(f"Mount path will be {os.getcwd()}/{Config.GIRDER_MOUNT_FOLDER}")
    try:
        girder_mounter.mount(path=Config.GIRDER_MOUNT_FOLDER, gc=gc)
    except Exception as e:
        print(f"Could not mount fuse volume: {e}")
    return jsonify({"message": "Success"}), 201

@app.route('/api/v1/unmount_client', methods=["POST"])
def unmount_girder_client():
    girder_mounter.unmount()
    return jsonify({"message": "Success"}), 200
    


# =============================================================================
# Flask Routes
# =============================================================================

# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def serve_react_app(path):
#     if path != '' and os.path.exists(os.path.join(app.static_folder, path)):
#         print("received request for", path, "returning static folder with path ", os.path.join(app.static_folder, path))
#         return send_from_directory(app.static_folder, path)
#     else:
#         print("received request for", path, "returning index.html from", app.static_folder)
#         return send_from_directory(app.static_folder, 'index.html')


# =============================================================================
# Dataset Wrangling
# =============================================================================
@app.route('/api/v1/select-dataset', methods=['POST'])
def select_dataset():
    new_study_name = request.form.get('new_study_name')
    source_study_name = request.form.get('source_study')
    new_study_id = request.form.get('new_study_id')
    print(f'{new_study_name=}')
    print(f'{source_study_name=}')
    # TODO: Validate source existance
    # TODO: Sanitize filenames
    # new_study_path = str(Path(app.config['DATASET_FOLDER'], new_study_name))
    # source_study_path = source_study_name

    new_folder = copy_folder_girder(gc, source_study_name, new_study_id)
    print(f"{new_folder=}")
    # # Make symlink to study TODO: Replace this with abstract file api
    # os.symlink(source_study_path, new_study_path, True)

    return Response(status=204)

@app.route('/api/v1/check-study-name', methods=['POST'])
def check_study_name():
    study_name = request.get_json().get('study_name')
    print(study_name)
    study_path = Path(Config.STUDIES_FOLDER, study_name)
    if study_path.exists():
        return jsonify({'exists': True})
    return jsonify({'exists': False})


# =============================================================================
# Train Model
# =============================================================================




# Training Page which calls the training function in the background
@app.route('/api/v1/train-model/training/start', methods=['POST'])
def start_training():
    # Get data out of form
    num_epochs = int(session['epochs'])
    early_stopping = session['early_stopping']
    patience = session['patience'] or 0  # This can be set dynamically if needed
    patience = int(patience)
    model_name = session['study_name']
    study_name = session['study_name']
    study_id = session['study_id']
    feature_extractor = session['feature_extractor']

    task = session['task']
    validation_split = float(session['validation_split'])/100
    print("START TRAINING API")
    print(f"{num_epochs=}\n{early_stopping=}\n{patience=}\n{model_name=}\n{study_name=}\n{feature_extractor=}\n{task=}\n{validation_split=}")

    # Create paths to save data
    derived_data_directory = Path(Config.APPDATA_FOLDER, session['study_name'])
    model_save_path = Path(derived_data_directory, Config.MODELS_FOLDER)

    #Dataset is already on girder
    dataset_path = Path(Config.STUDIES_FOLDER, study_name, Config.DATASET_FOLDER, 'train')

    print(f"{dataset_path=}")
    # Creating folders for patches, stitches, features, models and attention data
    
    # print(f"patches_folder: {type(patches_folder)=}\n{patches_folder=}")
    training_time = time.time_ns()
    
    ## Getting list of all the slides
    slides_list = [f for f in dataset_path.rglob('*.*') if f.suffix in get_wsi_extensions() and f.is_file()]
    
    ## Patching
    girder_patches_folder = create_folder_girder(gc, parent_id=study_id, folder_name=Config.PATCHES_FOLDER, parent_type='collection')
    girder_stitches_folder = create_folder_girder(gc, parent_id=study_id, folder_name=Config.STITCHES_FOLDER, parent_type='collection')
    patch_files_directory, _ = start_patching(slides_list, derived_data_directory, patch_size=256, dataset_role='train')
    ## Add to girder    
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.PATCHES_FOLDER), destination_id=girder_patches_folder['_id'], destination_type="folder")
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.STITCHES_FOLDER), destination_id=girder_stitches_folder['_id'], destination_type="folder")
    print(f"Importing Patches {resp=}")


    ## Feature Extraction
    girder_features_folder = create_folder_girder(gc, parent_id=study_id, folder_name=Config.FEATURE_DATA_FOLDER, parent_type='collection')
    feature_files, feature_map_size = start_feature_extraction(slides_list, derived_data_directory, feature_extractor, patch_files_directory, dataset_role=True)
    ## Add to girder
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.FEATURE_DATA_FOLDER), destination_id=girder_features_folder['_id'], destination_type="folder")
    print(f"Importing Features {resp=}")

    ## Training
    # models_folder = create_folder_girder(gc, parent_id=study_id, folder_name="models", parent_type='collection')
    training_data, train_indices, val_indices = start_model_training(feature_files, feature_map_size, model_save_path, num_epochs, early_stopping, patience, validation_split, feature_extractor=feature_extractor)

    # Correlate filepaths to girder items
    girder_resource_list = [gc.resourceLookup(slide.relative_to(Config.GIRDER_MOUNT_FOLDER)) for slide in slides_list]
    print(f"{girder_resource_list=}")

    final_train_predictions = training_data["final_train_predictions"]
    final_val_predictions = training_data["final_val_predictions"]
    print(f'{final_train_predictions=}')
    print(f'{final_val_predictions=}')
    for preds, girder_data in zip(final_train_predictions, [girder_resource_list[i] for i in train_indices]):
        preds['girderData'] = girder_data
    for preds, girder_data in zip(final_val_predictions, [girder_resource_list[i] for i in val_indices]):
        preds['girderData'] = girder_data

    model_details = {"model_name": model_name,
                     "feature_extractor": feature_extractor,
                     "classifier": "Default",
                     "epochs": num_epochs,
                     "early_stopping": early_stopping,
                     "timestamp": training_time,
                     "val_losses": training_data["val_losses"],
                     "val_accuracies": training_data["val_accuracies"],
                     "train_losses": training_data["train_losses"],
                     "train_accuracies": training_data["train_accuracies"],
                     "final_train_predictions": final_train_predictions,
                     "final_val_predictions": final_val_predictions,
                     }
    with open(Path(model_save_path, f'model_details.json'), 'w') as file:
        json.dump(model_details, file)
    
    girder_models_folder = create_folder_girder(gc, parent_id=study_id, folder_name=Config.MODELS_FOLDER, parent_type='collection')
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.MODELS_FOLDER), destination_id=girder_models_folder['_id'], destination_type="folder")
    print(f"Importing Models {resp=}")

    return '', 204


@app.route('/api/v1/testing/start', methods=['POST'])
def start_testing():
    # get a bunch of session data
    dataset_name=session['study_name']
    dataset_role=session['test_dataset_role']
    study_id = session['study_id']

    # get path to dataset/derived data
    dataset_path = Path(Config.STUDIES_FOLDER, dataset_name, Config.DATASET_FOLDER, dataset_role)
    print(f"{dataset_path=}")
    derived_data_directory = Path(Config.APPDATA_FOLDER, session['study_name'])

    # get path to model
    model_save_path = Path(derived_data_directory, Config.MODELS_FOLDER)
    with open(Path(model_save_path, "model_details.json")) as model_details_file:
        model_details = json.load(model_details_file)

    # get list of slides
    slides_list = [f for f in dataset_path.rglob('*.*') if f.suffix in get_wsi_extensions() and f.is_file()]
    print(f"{slides_list=}")

    # Correlate filepaths to girder items
    girder_resource_list = [gc.resourceLookup(slide.relative_to(Config.GIRDER_MOUNT_FOLDER)) for slide in slides_list]
    print(f"{girder_resource_list=}")

    # make sure patches and stitches folders exist
    girder_patches_folder = create_folder_girder(gc, parent_id=study_id, folder_name=Config.PATCHES_FOLDER, parent_type='collection')
    girder_stitches_folder = create_folder_girder(gc, parent_id=study_id, folder_name=Config.STITCHES_FOLDER, parent_type='collection')

    # Perform patching and then import results into girder
    patch_files_directory, stitch_directory = start_patching(slides_list, derived_data_directory, patch_size=256, dataset_role='test')
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.PATCHES_FOLDER), destination_id=girder_patches_folder['_id'], destination_type="folder")
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.STITCHES_FOLDER), destination_id=girder_stitches_folder['_id'], destination_type="folder")

    # Make features folder, perform extraction, import results into girder
    girder_features_folder = create_folder_girder(gc, parent_id=study_id, folder_name=Config.FEATURE_DATA_FOLDER, parent_type='collection')
    feature_files, feature_map_size = start_feature_extraction(slides_list, derived_data_directory,  feature_extractor=model_details['feature_extractor'], patch_files_directory=patch_files_directory, dataset_role='test')
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.FEATURE_DATA_FOLDER), destination_id=girder_features_folder['_id'], destination_type="folder")

    print(f"Uploading patches: {resp=}")
    # Testing will generate attention
    attention_results_folder = create_folder_girder(gc, parent_id=study_id, folder_name=Config.ATTENTION_RESULTS_FOLDER, parent_type='collection')
    reports, accuracy = start_model_testing(feature_files=feature_files, model_save_path=model_save_path)
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, "attention_results"), destination_id=attention_results_folder['_id'], destination_type="folder")
    print(f"Model testing: {resp}")

    print(f"{reports=}")
    
    # Add girder ids (and other data) to reports for future lookup
    for report, girder_data in zip(reports, girder_resource_list):
        report['girderData'] = girder_data

    # Save Prediction Report
    report_save_path = Path(model_save_path, 'model_test_results.json')

    #Get previous json if it exists
    if(report_save_path.exists()):
        print('Loading previous predictions')
        with open(report_save_path, 'r') as f:
            pred_data = json.load(f)
    else:
        print('No previous predictions found, new file will be created')
        pred_data = {}

    # Create section for dataset if it does not already exist
    if dataset_name not in pred_data:
        pred_data[dataset_name] = {}

    # Save reports under [dataset_name][dataset_role]
    # This will replace previous tests on the same data but the models should be deterministic by this point
    pred_data[dataset_name][dataset_role] = {'data':reports, 'timestamp_ns': time.time_ns(), 'accuracy': accuracy}

    # Save json
    with open(report_save_path, 'w') as outf:
        json.dump(pred_data, outf)

    girder_models_folder = create_folder_girder(gc, parent_id=study_id, folder_name=Config.MODELS_FOLDER, parent_type='collection')
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.MODELS_FOLDER), destination_id=girder_models_folder['_id'], destination_type="folder")
    return '', 204

@app.route('/api/v1/inference/start', methods=['POST'])
def start_inference():
    request_json = request.get_json()
    filenames = request_json['fileNames']

    # path to images for inference
    images_path = Path(Config.STUDIES_FOLDER, session['study_name'], Config.DATASET_FOLDER, 'inference')

    # get specific image(s) for this inference
    slides_list = [Path(images_path, n) for n in filenames]

    # Correlate filepaths to girder items
    girder_resource_list = [gc.resourceLookup(slide.relative_to(Config.GIRDER_MOUNT_FOLDER)) for slide in slides_list]
    print(f"{girder_resource_list=}")

    # get path to dataset/derived data
    dataset_path = Path(Config.STUDIES_FOLDER, session['study_name'], Config.DATASET_FOLDER, 'inference')
    print(f"{dataset_path=}")
    derived_data_directory = Path(Config.APPDATA_FOLDER, session['study_name'])

    # get path to model
    model_save_path = Path(derived_data_directory, Config.MODELS_FOLDER)
    with open(Path(model_save_path, "model_details.json")) as model_details_file:
        model_details = json.load(model_details_file)

    # make sure patches and stitches folders exist
    girder_patches_folder = create_folder_girder(gc, parent_id=session['study_id'], folder_name=Config.PATCHES_FOLDER, parent_type='collection')
    girder_stitches_folder = create_folder_girder(gc, parent_id=session['study_id'], folder_name=Config.STITCHES_FOLDER, parent_type='collection')

    # Perform patching and then import results into girder
    patch_files_directory, stitch_directory = start_patching(slides_list, derived_data_directory, patch_size=256, dataset_role='inference')
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.PATCHES_FOLDER), destination_id=girder_patches_folder['_id'], destination_type="folder")
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.STITCHES_FOLDER), destination_id=girder_stitches_folder['_id'], destination_type="folder")

    # Make features folder, perform extraction, import results into girder
    girder_features_folder = create_folder_girder(gc, parent_id=session['study_id'], folder_name=Config.FEATURE_DATA_FOLDER, parent_type='collection')
    feature_files, feature_map_size = start_feature_extraction(slides_list, derived_data_directory,  feature_extractor=model_details['feature_extractor'], patch_files_directory=patch_files_directory, dataset_role='inference')
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, Config.FEATURE_DATA_FOLDER), destination_id=girder_features_folder['_id'], destination_type="folder")

    # Testing will generate attention
    attention_results_folder = create_folder_girder(gc, parent_id=session['study_id'], folder_name=Config.ATTENTION_RESULTS_FOLDER, parent_type='collection')
    reports, accuracy = start_model_testing(feature_files=feature_files, model_save_path=model_save_path)
    resp = import_data_to_assetstore_girder(gc, assetstore_name="Assetstore", assetstore_id=None, source_data_path=Path(derived_data_directory, "attention_results"), destination_id=attention_results_folder['_id'], destination_type="folder")

    # Add girder ids (and other data) to reports for future lookup
    for report, girder_data in zip(reports, girder_resource_list):
        report['girderData'] = girder_data
    print(reports)
    
    # discard labels as they do not do anything in this context
    for report in reports:
        del report['label']
    socketio.emit('inference-update', {'report': reports})

    return Response(status=204)

## TODO: Check the validity of all the session variables before initiating training/patching
## TODO: Add option to work with train/test and any different dataset
## TODO: Add support for different FEs.
## TODO: Throw/Catch an error if entered FE is not found
## TODO: Remove the `suppress_output` thing

def start_patching(slides_list, base_directory, patch_size=256, dataset_role='train'):
    print("***********************Patching")
    ## TODO: Make sure the skip_existing function checks in the girder directory
    ##          and not in the temp directory for duplicates/pre-existing patches
    # slides_list = [f for f in dataset_path.rglob('*.*') if f.suffix in get_wsi_extensions()]
    patch_files_list = []
    # print(f"PATCHING::\n\n{slides_list=}")

    # Base path: derived_data/{study}
    # directory: derived_data/{study}/patches/{train/test}/{size}/{label}/file
    patch_directory = Path(base_directory, Config.PATCHES_FOLDER, dataset_role, f"size_{patch_size}")
    stitch_directory = Path(base_directory, Config.STITCHES_FOLDER, dataset_role, f"size_{patch_size}")
    # print(f"{patch_directory=}\n{stitch_directory}")
    # t0 = time.time()

    for idx, wsi_file in enumerate(slides_list):
        patch_save_path = Path(patch_directory ,wsi_file.parent.stem, wsi_file.with_suffix(".h5").name)
        stitch_save_path = Path(stitch_directory, wsi_file.parent.stem, wsi_file.with_suffix(".jpg").name)
        # print(f"{patch_save_path=}\n{stitch_save_path=}")
        # print(f"{patch_save_path=}\n{stitch_save_path}")
        # print(f"Relative path: {wsi_file.relative_to(dataset_path)}")

        patch_file_path, stitch_file_path, pre_existing = patch(wsi_file, patch_size=patch_size,
                                                                    patch_save_path=patch_save_path, 
                                                                    stitch_save_path=stitch_save_path,
                                                                    skip_existing=True, )
        patch_files_list.append(patch_file_path)
        socketio.emit('patching-update', {'progress_total':len(slides_list),
                    'progress_done':idx+1})
    # upload_folder_to_girder(gc, Path(Config.TEMP_FOLDER, 'patches'), girder_folder_id)
    # t1 = time.time()
    # print(f'Time elapsed: {t1-t0}')

    return patch_directory, stitch_directory

def start_feature_extraction(slides_list, base_directory, feature_extractor, patch_files_directory, dataset_role='train'):
    print("***********************Extracting")
    features_directory = Path(base_directory, Config.FEATURE_DATA_FOLDER)
    extractor = Extractor(model_name=feature_extractor)
    feature_files_list = []
    # feature_extractor='hibou_b'
    # print(f"{patch_files_directory=}")
    # print(f"{features_directory=}")
    patch_directory = Path(Config.APPDATA_FOLDER,session['study_name'],Config.PATCHES_FOLDER)
    for idx, wsi_file in enumerate(slides_list):
        parent_directory = Path(*wsi_file.parts[-2:])
        patch_path = Path(patch_files_directory, parent_directory).with_suffix(".h5")
        feature_save_path = Path(features_directory, feature_extractor, patch_path.relative_to(patch_directory)).with_suffix(".h5")
        # print(f"{parent_directory=}")
        # print(f"{patch_path=}")
        # print(f"{feature_save_path=}")
        feature_file_path, pre_existing = extractor.extract(wsi_file, feature_save_path=feature_save_path, patch_file_path=patch_path, skip_existing=True)
        feature_files_list.append(feature_file_path)
        progress = {'progress_total':len(slides_list),
                    'progress_done':idx+1}
        socketio.emit('extract-update', progress)

    feature_map_size = extractor.get_feature_map_size()
    return feature_files_list, feature_map_size

# Callback function to send to 'pooler' so Model can send live updates while training/testing
def send_update_socketio(update_event, update_message):
    socketio.emit(update_event, update_message)

def start_model_training(feature_files, feature_map_size, model_save_path: Path, num_epochs, early_stopping, patience, validation_split, feature_extractor):
    # print(f"{model_save_path=}")
    trainer = TransMILPooler(num_epochs=num_epochs, input_size=feature_map_size, random_state=42, early_stopping=early_stopping, patience=patience, val_split=validation_split, callback_fn=send_update_socketio)
    model, class_map, progress, train_indices, val_indices = trainer.train(feature_files, feature_extractor)
    model_save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model, Path(model_save_path, f'{Config.CLASSIFIER_NAME}'))
    with open(Path(model_save_path, f'class_map.json'), "w") as outfile:
        outfile.write(json.dumps(class_map))

    socketio.emit('training-complete')
    return progress, train_indices, val_indices

def start_model_testing(feature_files, model_save_path):
    print("***********************Testing")
    tester = TransMILPooler(model = Path(model_save_path, f'{Config.CLASSIFIER_NAME}'), callback_fn=send_update_socketio)
    

    predictions, accuracy = tester.test(features=feature_files)  
    socketio.emit('testing-complete')
    
    return predictions, accuracy 

# =============================================================================
# APIs
# =============================================================================


@app.route('/api/v1/model-details', methods=['GET'])
def get_model_details():
    model_name = request.args.get("model_name")
    model_metadata_path = Path(Config.STUDIES_FOLDER, model_name, Config.MODELS_FOLDER, "model_details.json")
    model_details = json.load(open(model_metadata_path))
    # This is too long to be printing like this
    # print(f"returned model details from api/model-details:\n{model_details=}")
    return jsonify(model_details)


@app.route('/api/v1/datasets/info', methods=['GET'])
def get_dataset_info_json():
    study_name = request.args.get('ds_name')
    dataset_role = request.args.get('ds_role')
    # print(f".../datasets/info API: {study_name=}, {dataset_role=}")
    dataset_path = Path(Config.STUDIES_FOLDER, study_name, Config.DATASET_FOLDER, dataset_role)
    # print(dataset_path)
    info = helper.get_dataset_basic_info(dataset_path)
    return Response(info.to_json(orient="records"), mimetype='application/json')

@app.route('/api/v1/datasets/list')
def list_datasets_api():
    ds_dir = Path(Config.STUDIES_FOLDER)
    dataset_names = filter(lambda d: Path(ds_dir, d).is_dir(), os.listdir(ds_dir))
    result = {}
    # print(f"{list(dataset_names)}")
    for n in dataset_names:
        # result[n] = list(filter(lambda d: Path(ds_dir, n, d).is_dir(), os.listdir(Path(ds_dir, n, "datasets"))))
        result[n] = ["train", "test"]

    return jsonify(result)

@app.route('/api/v1/collections')
def list_collections():
    collections = gc.listCollection()
    result = [{'id':c['_id'], 'name':c['name']} for c in list(collections) if not c['meta'].get('hiddenFromUser', False)]
    print(f'{result=}')
    return jsonify(result)

@app.route('/api/v1/dataset/wsi/thumbnail', methods=['GET'])
def get_wsi_thumb_api():
    filepath = request.args.get('path')
    print(f'Got request for thumbnail of {filepath}')
    slide = openslide.open_slide(filepath)
    thumb = slide.get_thumbnail((1000, 1000))
    return helper.serve_pil_image(thumb)

@app.route('/api/v1/results/attention/image')
def get_attention_image():
    filepath = request.args.get('path')
    print(f'attention img requested: {filepath}')
    ## FIXME: fix filepath doubling `backend` directory
    filepath = filepath.replace("backend/","")
    # filepath =
    print(f"After update {filepath}")
    return send_file(filepath) # Convert to send from directory

@app.route('/api/v1/results/')
def get_test_results_api():
    # get args
    model_name = request.args.get('model_name')
    ds_name = request.args.get('ds_name')
    ds_role = request.args.get('ds_role')

    # get predictions
    # pred_path = Path(Config.MODELS_FOLDER, model_name, 'model_test_results.json')
    pred_path = Path(Config.STUDIES_FOLDER, model_name, Config.MODELS_FOLDER, 'model_test_results.json')
    print(f"{pred_path=}")
    with open(pred_path, 'r') as f:
        j = json.load(f)
        predictions = j[ds_name][ds_role]['data']
        timestamp = j[ds_name][ds_role]['timestamp_ns']

    # get label map
    map_path = Path(Config.STUDIES_FOLDER, model_name, Config.MODELS_FOLDER, 'class_map.json')
    with open(map_path, 'r') as f:
        label_idx_map = json.load(f)
        
    result = {
        'map':label_idx_map,
        'predictions':predictions,
        'timestamp_ns':timestamp
    }

    return jsonify(result)

@app.route('/api/v1/model/classmap/<model_name>')
def get_model_classmap(model_name):
    # get label map
    map_path = Path(Config.STUDIES_FOLDER, model_name, Config.MODELS_FOLDER, 'class_map.json')
    with open(map_path, 'r') as f:
        label_idx_map = json.load(f)
    return jsonify(label_idx_map)

@app.route('/api/v1/models/list')
def get_model_list_api():
    models_list = get_models_list()
    return jsonify(models_list)

# TODO: remove the sample results dict and throw/return an error
@app.route('/api/v1/models/details/<model_name>')
def get_model_details_api(model_name):
    model_details_path = Path(Config.STUDIES_FOLDER, model_name, Config.MODELS_FOLDER, "model_details.json")
    model_test_path = Path(Config.STUDIES_FOLDER, model_name, Config.MODELS_FOLDER, "model_test_results.json")
    model_details = {}
    if model_details_path.exists():
        model_details = json.load(open(model_details_path))

    test_results = []
    if model_test_path.exists():
        test_results_json = json.load(open(model_test_path))
        for dataset_name, dataset_info in test_results_json.items():
            for role, role_info in dataset_info.items():
                test_results.append({
                    "dataset_name": dataset_name,
                    "dataset_role": role,
                    "timestamp": role_info["timestamp_ns"],
                    "accuracy": role_info["accuracy"]
                })
    return jsonify(model_details, test_results)

# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    socketio.run(app=app, 
                 host='0.0.0.0', 
                 port=8000, 
                 debug=True, 
                 allow_unsafe_werkzeug=True)


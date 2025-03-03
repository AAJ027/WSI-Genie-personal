from pathlib import Path

def find_folder_id(gc, folder_name, collection_id):
    ## MAYBE It will be easier if I can make frontend/plugin return the 
    ## folder id for datasets folder

    # search for the folder within this collection
    folders = list(gc.listFolder(collection_id, name=folder_name, parentFolderType='collection'))
    if not folders:
        raise ValueError(f"Folder '{folder_name}' not found in collection with id '{collection_id}'")
    return folders[0]['_id']

def get_collection_id(gc, collection_name):
    print(f"{collection_name}")
    collection_list = gc.get("collection", {"text": collection_name})
    print(f"{collection_list=}\t{len(collection_list)=}")
    
    if len(collection_list) != 1:
        raise ValueError(f"Error fetching Collection '{collection_name}'")
    return collection_list[0]['_id']

def get_folder_girder(gc, folder_id):
    try:
        return gc.getFolder(folder_id)
    except Exception as e:
        print(f"Error getting folder: {str(e)}")
        return None

def get_collection_girder(gc, collection_id):
    try:
        return gc.getCollection(collection_id)
    except Exception as e:
        print(f"Error getting collection: {str(e)}")
        return None

def upload_folder_to_girder(gc, local_path: Path, parent_folder_id: str):
    for item in local_path.iterdir():
        if item.is_file():
            print(f"Uploading file: {item}")
            gc.uploadFileToFolder(parent_folder_id, str(item))
        elif item.is_dir():
            print(f"Creating folder: {item.name}")
            new_folder = create_folder_girder(gc, parent_folder_id, item.name)
            upload_folder_to_girder(gc, item, new_folder['_id'])

def create_folder_girder(gc, parent_id, folder_name, parent_type='folder'):
    try:
        new_folder = gc.createFolder(parent_id, folder_name, parentType=parent_type, reuseExisting = True)
        print(f"Folder '{folder_name}' created successfully.")
        return new_folder
    except Exception as e:
        print(f"Error creating folder: {str(e)}")
        return None

def get_assetstore_id(gc, assetstore_name):
    asset_list = gc.get(path='/assetstore')
    for asset_details in asset_list:
        if asset_details['name'] == assetstore_name:
            return asset_details['_id']
    print(f"Asseststore {assetstore_name} not found!")
    return None

def import_data_to_assetstore_girder(gc, assetstore_name, assetstore_id, source_data_path, destination_id, destination_type):
    # Remove .DS_Store file from root if it exists
    Path(source_data_path, ".DS_Store").unlink(missing_ok=True)
    
    if not assetstore_id:
        assetstore_id = get_assetstore_id(gc, assetstore_name)
    parameters = {'importPath': f"/{source_data_path}",
                  'destinationId': destination_id,
                  'destinationType': destination_type,
                 }
    response = gc.post(f"assetstore/{assetstore_id}/import", parameters=parameters)
    
    return response

def copy_folder_girder(gc, source_study_name, new_study_id):
    print(f"{source_study_name=}\n{new_study_id=}")
    source_study_id = get_collection_id(gc, source_study_name)
    datasets_folder_id = find_folder_id(gc, "datasets", collection_id=source_study_id)
    source_folder = get_folder_girder(gc, datasets_folder_id)
    new_study = get_collection_girder(gc, new_study_id)
    print(f"Starting copy")
    print(f"Source collection {source_study_id}, new collection {new_study_id}")
    request_params = {"parentType": "collection",
                      "parentId": new_study_id,
                      "name":"datasets"}
    copy_response = gc.post(f"folder/{source_folder['_id']}/copy", parameters=request_params)
    # copied_folder = Folder().copyFolder(source_folder, parent=new_study, name="datasets", parentType = "collection")
    # copied_folder_2 = Folder().copyFolderComponents(source_folder, copied_folder, None, None, None)

    print(f"{copy_response=}")
    return copy_response

# Example usage and testing
if __name__ == '__main__':
    import girder_client

    # This part is just for testing - you wouldn't include this in your server code
    gc = girder_client.GirderClient()
    gc.setToken("xmIGt8Gmq04pnl45pWFLRXXZ7xCPfMaqa9e8ejXRV0qwZdSgr1qeM6wE6vDqFyMl")

    # Find the source folder ID
    SOURCE_FOLDER_NAME = 'datasets'
    PARENT_COLLECTION_ID = '66f2bbcc8ed4bdadcc7c4121'
    SOURCE_FOLDER_ID = find_folder_id(gc, SOURCE_FOLDER_NAME, PARENT_COLLECTION_ID)

    DESTINATION_FOLDER_ID = '66f2f69d8ed4bdadcc7c41c5'

    # Test the new functions
    folder = get_folder_girder(gc, SOURCE_FOLDER_ID)
    if folder:
        print(f"Successfully retrieved folder: {folder['name']}")

    collection = get_collection_girder(gc, PARENT_COLLECTION_ID)
    if collection:
        print(f"Successfully retrieved collection: {collection['name']}")

    # Test finding collection ID by name
    collection_name = "Test Collection"
    try:
        collection_id = find_collection_id(gc, collection_name)
        print(f"Found collection ID for '{collection_name}': {collection_id}")
    except ValueError as e:
        print(str(e))

    # copy_directory_tree_girder(gc, SOURCE_FOLDER_ID, DESTINATION_FOLDER_ID)
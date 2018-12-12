import os
import os.path

def asset_paths_from(saved_model_dir):
    asset_paths = {}
    asset_path_basedir = os.path.join(saved_model_dir, 'assets.extra')
    for root, dirs, files in os.walk(asset_path_basedir):
        root = os.path.abspath(root)
        for filename in files:
            filepath = os.path.join(root, filename)
            relpath = os.path.relpath(filepath, asset_path_basedir)
            asset_paths[relpath] = filepath

    return asset_paths
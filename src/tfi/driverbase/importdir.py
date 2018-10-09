import tempfile
import zipfile
import os
import time
import json

def _maybe_squashfs_image(file):
  try:
    import PySquashfsImage
    return PySquashfsImage.SquashFsImage(file)
  except ImportError:
      return None
  except IOError:
    return None

def import_dir(mktempdir, import_path):
  if os.path.isdir(import_path):
    return import_path

  import zipfile
  if zipfile.is_zipfile(import_path):
    import_dir = mktempdir()
    import subprocess
    print("Trying fuse-zip")
    result = subprocess.run(["fuse-zip", import_path, import_dir])
    if result.returncode:
      print(result.stderr or result.stdout)
      print("Falling back to extracting zip after fuse-zip failed.")
      with zipfile.ZipFile(import_path) as zipf:
        zipf.extractall(import_dir)

    while True:
      extracted_entries = os.listdir(import_dir)
      if len(extracted_entries) == 1:
        return os.path.join(import_dir, extracted_entries[0])
  
  squashfs_image = _maybe_squashfs_image(import_path)
  if squashfs_image:
    squashfs_image.close()
    import subprocess
    import_dir = mktempdir()

    # result = subprocess.run(["mount", import_path, import_dir, "-t", "squashfs", "-o", "loop,ro,noexec,noload"])
    result = subprocess.run(["squashfuse", import_path, import_dir])
    if result.returncode:
      raise Exception(result.stderr or result.stdout)

    return import_dir

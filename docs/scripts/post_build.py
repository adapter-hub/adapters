import os
import shutil
import sys


BUILD_DIR = sys.argv[1]

for folder in os.listdir(BUILD_DIR):
    path = os.path.join(BUILD_DIR, folder)
    if folder == "main":
        file_names = os.listdir(path)
        for file_name in file_names:
            shutil.move(os.path.join(path, file_name), BUILD_DIR)
        os.rmdir(path)
    else:
        shutil.move(path, path.replace("adapters", "v"))

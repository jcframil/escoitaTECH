import os
from pathlib import Path

def load_route():
    out_path = Path("./audio/")
    dir = os.listdir(out_path)
    routes_namefiles=dict()
    for file in dir:
        name_file = file
        routes_namefiles[out_path/name_file] = name_file
    return routes_namefiles

def check_folder(name_folder):
    out_path = Path("./figures")
    out_path = out_path / name_folder
    out_path.mkdir(parents=True, exist_ok=True)
    return  out_path




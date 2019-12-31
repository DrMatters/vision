import pathlib
import os
import json
import argparse
from pprint import pprint


def create_relative_walk(directory):
    res = []
    for dirpath, dirnames, filenames in os.walk(directory):
        cur_dir = pathlib.Path(dirpath)
        for filename in filenames:
            full = cur_dir / filename
            rel_path = full.relative_to(directory).as_posix()
            res.append(rel_path)
    return res


def compare_dir_with_walk(walk, directory):
    walk = set(walk)
    dir_walk = set(create_relative_walk(directory))
    not_present_in_walk = dir_walk - walk
    not_present_in_dir = walk - dir_walk
    return not_present_in_walk, not_present_in_dir


def main():
    parser = argparse.ArgumentParser(description='Compares folder structure with saved walk')
    parser.add_argument("--stored_walk", default='./res_walk.json', type=str, help="Location of .json file containing "
                                                                                   "saved file structure")
    parser.add_argument("--folder", required=True, type=str, help="Location of folder containing file structure to "
                                                                  "compare with stored walk")
    args = parser.parse_args()
    stored_walk_location = args.stored_walk

    with open(stored_walk_location, 'r') as f:
        walk = json.load(f)

    not_present_in_walk, not_present_in_folder = compare_dir_with_walk(walk, args.folder)
    print('Files not present in saved walk:')
    pprint(list(not_present_in_walk))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Files not present in local folder:')
    pprint(list(not_present_in_folder))
    if not not_present_in_folder and not not_present_in_walk:
        print('Folders are equal')


if __name__ == '__main__':
    main()

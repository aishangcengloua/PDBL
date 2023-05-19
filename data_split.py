import numpy
import argparse
import shutil
import codecs
from pathlib import Path

def dataset_size(path) :
    parents_dirs = [name for name in Path(path).iterdir() if name.is_dir()]
    total_numbers = 0
    for dir in parents_dirs:
        for classes in dir.iterdir():
            number = 0
            for _ in classes.iterdir():
                number += 1
            total_numbers += number
    print(total_numbers)

def move_file(src_path, train_dst_path, test_dst_path, rate) :
    parents_dirs = [name for name in Path(src_path).iterdir() if name.is_dir()]
    for dir in parents_dirs :
        for classes in dir.iterdir() :
            print(classes)
            number = 0
            for _ in classes.iterdir() :
                number += 1

            for idx, name in enumerate(classes.iterdir()) :
                src_split = name.__str__().split('\\')
                img_name = src_split[-1]
                src_split.pop()
                if idx < int(rate * number) :
                    src_split[1] = train_dst_path.split('/')[1]
                else :
                    if test_dst_path is not None :
                        src_split[1] = test_dst_path.split('/')[1]
                    else :
                        break

                dst_path = "/".join(src_split)
                dst_path = Path(dst_path)
                if not dst_path.exists() :
                    dst_path.mkdir(parents = True, exist_ok = True)

                dst_path = dst_path.joinpath(img_name)
                shutil.copyfile(name, dst_path)

if __name__ == '__main__':
    # move_file('PDBL_Dataset/LC25000', 'PDBL_Dataset/LC_Train_100', 'PDBL_Dataset/LC_Test', 0.6)
    move_file('PDBL_Dataset/LC_Train_100', 'PDBL_Dataset/LC_Train_001', None, 0.01)
    dataset_size('PDBL_Dataset/LC25000')
    dataset_size('PDBL_Dataset/LC_Train_100')
    dataset_size('PDBL_Dataset/LC_Test')
    dataset_size('PDBL_Dataset/LC_Train_001')

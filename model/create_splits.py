import os
import argparse
import glob
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data", help="path to data folder to create splits from")
parser.add_argument("--splits_path", type=str, default=None, help="path to data folder to create splits in")
parser.add_argument("--train_pct", type=int, default=90, help="percentage of samples for training")
parser.add_argument("--valid_pct", type=int, default=5, help="percentage of samples for validation")
parser.add_argument("--test_pct", type=int, default=5, help="percentage of samples for testing")

def prepare_folders(args):
    splits_path = args.splits_path
    if splits_path is None:
        # data_path = os.path.dirname(os.path.abspath(__file__)) if not os.path.isabs(args.data_path) else data_path
        splits_path = os.path.dirname(os.path.abspath(args.data_path))
    print "Creating splitted dataset in", splits_path
    if not os.path.exists(splits_path):
        os.makedirs(splits_path)
    train_dir = os.path.join(splits_path, 'splits', 'train')
    valid_dir = os.path.join(splits_path, 'splits', 'valid')
    test_dir = os.path.join(splits_path, 'splits', 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    return train_dir, valid_dir, test_dir

# check, if file exists, make link
def check_link(in_file, out_dir):
    basename = os.path.basename(in_file)
    if os.path.exists(in_file):
        link_file = os.path.join(out_dir, basename)
        rel_link = os.path.relpath(in_file, out_dir)
        os.symlink(rel_link, link_file)

def create_splits(args):
    files = glob.glob(args.data_path + "/*")
    files_df = pd.DataFrame(map(lambda f: (f,) + os.path.splitext(os.path.basename(f)), files), columns=["filename", "sample", "ext"])
    files_df['sample'] = files_df['sample'].apply(lambda x: x.replace(".pickle", ""))
    files_df.set_index("sample", inplace=True)
    sample_names = files_df.index.unique().values
    frac_train = float(args.train_pct) / 100
    frac_valid = float(args.valid_pct) / 100
    frac_test  = float(args.test_pct) / 100
    len_splits = [int(frac_train*len(sample_names)), int((frac_train+frac_valid)*len(sample_names))]
    np.random.shuffle(sample_names)
    train_idx, valid_idx, test_idx = np.split(sample_names, len_splits)
    train_files = files_df.loc[train_idx]['filename'].values
    valid_files = files_df.loc[valid_idx]['filename'].values
    test_files  = files_df.loc[test_idx]['filename'].values
    return train_files, valid_files, test_files

def add_splits(args):
    train_files, valid_files, test_files = create_splits(args)
    train_dir, valid_dir, test_dir = prepare_folders(args)
    for f in train_files:
        check_link(f, train_dir)
    for f in test_files:
        check_link(f, test_dir)
    for f in valid_files:
        check_link(f, valid_dir)

if __name__ == '__main__':
    # python create_splits.py --train_pct=100 --valid_pct=0 --test_pct=0 --data_path=/home/data/world-cities/spatial-maps/samples --splits_path=/home/data/world-cities/spatial-maps/
    args, unparsed = parser.parse_known_args()
    add_splits(args)

import os, glob
from math import *
from tqdm import tqdm
import shutil

input_folders = ['datasets/unsorted']

BASE_DIR_ABSOLUTE = "C:\\git-projects\\Detect-fall\\"
OUT_DIR = './datasets/prepared-dataset/'

OUT_TRAIN = f'{OUT_DIR}train/'
OUT_VAL = f'{OUT_DIR}val/'

coeff = [80, 20]
exceptions = ['classses']

if int(coeff[0] + int(coeff[1]) > 100):
    print("Coeff can't exceed 100%")
    exit(1)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


print(f"Preparing images data by {coeff[0]}/{coeff[1]} rule.")
print(f"Source folders: {len(input_folders)}")
print("Gathering data ...")


source = {}
for sf in input_folders:
    source.setdefault(sf, [])

    os.chdir(BASE_DIR_ABSOLUTE)
    os.chdir(sf)

    for filename in glob.glob("*.jpg"):
        source[sf].append(filename)


train = {}
val = {}
chunks = 10
for sk, sv in source.items():
    train_chunk = floor(chunks * (coeff[0] / 100))
    val_chunk = chunks - train_chunk

    train.setdefault(sk, [])
    val.setdefault(sk, [])
    for item in chunker(sv, chunks):
        train[sk].extend(item[:train_chunk])
        val[sk].extend(item[train_chunk:])


train_sum = sum(len(sv) for sv in train.values())
val_sum = sum(len(sv) for sv in val.values())
print(f"\nOverall TRAIN images count: {train_sum}")
print(f"\nOverall TEST images count: {val_sum}")

os.chdir(BASE_DIR_ABSOLUTE)
print("\nCopying TRAIN source items to prepared folder ...")
for sk, sv in tqdm(train.items()):
    for item in tqdm(sv):
        imgfile_source = sk + item
        imgfile_dest = OUT_TRAIN + sk.split('/')[-2] + '/'

        os.makedirs(imgfile_dest, exist_ok=True)
        shutil.copyfile(imgfile_source, imgfile_dest + item)


os.chdir(BASE_DIR_ABSOLUTE)
print("\nCopying VAL source items to prepared folder ...")
for sk, sv in tqdm(val.items()):
    for item in tqdm(sv):
        imgfile_source = sk + item
        imgfile_dest = OUT_VAL + sk.split('/')[-2] + '/'

        os.makedirs(imgfile_dest, exist_ok=True)
        shutil.copyfile(imgfile_source, imgfile_dest + item)


print("\nDONE!")
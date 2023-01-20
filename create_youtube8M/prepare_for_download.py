import os, tqdm, glob, shutil
import subprocess
import numpy as np

def read_txt_lines(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    return lines

def filter_accessdenie(tockens):
    tockens_ = list()
    for x in tockens:
        if x != 'AccessDenie':
            tockens_.append(x)
    return tockens_

def write_for_downloading(tockens, cat_name):
    sv_folder = 'todownload'
    os.makedirs(sv_folder, exist_ok=True)
    with open(os.path.join(sv_folder, '{}.txt'.format(cat_name)), 'w') as fp:
        for x in tockens:
            fp.write("%s\n" % x)

def check_existence(cat_name):
    file_path = os.path.join('todownload', '{}.txt'.format(cat_name))
    return os.path.exists(file_path)

# Read all Video Tags
filename = os.path.join('youtube8mcategories.txt')
lines = read_txt_lines(filename)

if os.path.exists('category-ids/'):
    shutil.rmtree('category-ids/')

cat_names, nvideos = list(), list()
for x in lines:
    code, name_nvideo = x.split('\t')

    nvideo = name_nvideo.split(' ')[-1]
    nvideo_strlen = len(nvideo)
    nvideo = int(nvideo[1:-1])

    cat_name = name_nvideo[0:-(nvideo_strlen + 1)]

    nvideos.append(nvideo)
    cat_names.append(cat_name)

# Reverse in Convenience of Debugging
nvideos = nvideos[::-1]
cat_names = cat_names[::-1]

count = 0
# Extract Necessary Tokens
for cat_name, _ in tqdm.tqdm(zip(cat_names, nvideos)):
    # cat_name = cat_names[1]
    if check_existence(cat_name):
        print("%s Finished" % cat_name)
        continue

    assert os.path.exists("downloadcategoryids.sh")
    with open('selectedcategories.txt', 'w') as f:
        f.write(cat_name)
    ret = subprocess.call("bash downloadcategoryids.sh %d %s" % (1e7, cat_name), shell=True)

    # Start to Post-Process the Downloaded File
    txt_path = glob.glob(os.path.join('category-ids/*.txt'))
    if len(txt_path) == 0:
        print("%s Failed" % (cat_name))
        continue
    a = 1
    assert len(txt_path) == 1
    tockens = read_txt_lines(txt_path[0])
    tockens_ = filter_accessdenie(tockens)
    write_for_downloading(tockens_, cat_name)

    shutil.rmtree('category-ids/')
    print("%s with %d vid Finished" % (cat_name, len(tockens_)))
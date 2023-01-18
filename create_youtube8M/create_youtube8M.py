import os, tqdm

# Num of paired images is 665583
total_num = 1281167 # <- Reduction

# Read all Video Tags
filename = os.path.join('youtube8mcategories.txt')
with open(filename) as file:
    lines = [line.rstrip() for line in file]

cat_names, nvideos = list(), list()
for x in lines:
    code, name_nvideo = x.split('\t')

    nvideo = name_nvideo.split(' ')[-1]
    nvideo_strlen = len(nvideo)
    nvideo = int(nvideo[1:-1])

    cat_name = name_nvideo[0:-(nvideo_strlen + 1)]

    nvideos.append(nvideo)
    cat_names.append(cat_name)

# Extract Necessary Tokens
for tqdm.tqdm(cat_names):
    a = 1
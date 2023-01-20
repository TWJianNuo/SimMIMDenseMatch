import cv2, copy, time
import os, tqdm, glob, shutil, random
import subprocess
import numpy as np
import yt_dlp, pafy, urllib
import PIL.Image as Image
from loguru import logger

logger.add("Youtube8MGeneration.log")

def read_txt_lines(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    return lines

def collect_all_vids(old_vids):
    maxnum = 100000
    new_vids = dict()

    download_fold = 'todownload'
    all_cats_token_files = glob.glob(os.path.join(download_fold, '*.txt'))

    for f in all_cats_token_files:
        cat_name = f.split('/')[-1]
        cat_name = cat_name.split('.')[0]

        tokens = read_txt_lines(f)

        if len(tokens) > maxnum:
            tokens = tokens[0:maxnum]

        if cat_name not in old_vids:
            new_vids[cat_name] = tokens
        old_vids[cat_name] = tokens

    return new_vids, old_vids

def get_pafy_url(url):
    url_data = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(url_data.query)
    id = query["v"][0]
    video = 'https://youtu.be/{}'.format(str(id))

    return video

sv_fold = '/media/shengjie/scratch2/EMAwareFlowDatasets/Youtube8M'
if os.path.exists(sv_fold):
    shutil.rmtree(sv_fold)
os.makedirs(sv_fold, exist_ok=True)

old_vids = dict()
total_num = 1281167 # <- Reduction
new_vids, old_vids = collect_all_vids(old_vids)

if len(list(new_vids.keys())) > 0:
    totnum = 0
    for x in new_vids.keys():
        logger.info("Generate %s Video %d" % (x, len(new_vids[x])))
        totnum += len(new_vids[x])

start_time = 35
sec_interval = [0, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 20]

succeed = 0
failure = 0

st = time.time()
for cat in new_vids.keys():
    all_tokens = new_vids[cat]

    cat_fold = os.path.join(sv_fold, cat)
    os.makedirs(cat_fold, exist_ok=True)

    for idx, token in enumerate(tqdm.tqdm(all_tokens, disable=True)):

        try:
            url = 'https://www.youtube.com/watch?v={}'.format(token)

            urlPafy = pafy.new(get_pafy_url(url))
            videoplay = urlPafy.getbest(preftype="any")

            cap = cv2.VideoCapture(videoplay.url)

            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            success, img = cap.read()
            if success:
                random_st = start_time
            else:
                # Get Duration
                ydl = yt_dlp.YoutubeDL({})
                info_dict = ydl.extract_info(url, download=False)
                fps = info_dict['fps']
                duration = info_dict['duration']
                random_st = np.random.randint(int(np.ceil(0.3 * duration)), int(np.floor(0.7 * duration)))
        except:
            failure += 1

        for iidx, sec in enumerate(sec_interval):
            csec = random_st + sec
            cap.set(cv2.CAP_PROP_POS_MSEC, csec * 1000)
            success, img = cap.read()

            nofold = True
            if success:

                if nofold:
                    token_fold = os.path.join(cat_fold, str(idx))
                    os.makedirs(token_fold, exist_ok=True)
                    nofold = False

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((192, 160))
                img.save(os.path.join(token_fold, '{}.jpg'.format(str(iidx))))

        succeed += 1

        if np.mod(succeed + failure, 100) == 0:
            resttime = (time.time() - st) / (succeed+failure) * (totnum - succeed - failure)
            resttime = resttime / 60 / 60
            logger.info("Processed %d / %d, remained hour %f, Fail Rate %f" % (succeed, totnum, resttime, failure / (succeed+failure)))
    break

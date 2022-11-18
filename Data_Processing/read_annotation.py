import json
import glob
import os
import re
import requests
import tqdm


vlnum = 0 #村の数
annonum = 0 #アノテーションの数
villageIDdict = {}

for file in glob.glob('./corpus\G*.json'):
    with open(file, 'r', encoding="utf-8") as f:
        jsn = json.load(f)
        annotations = list(jsn.values()) #要素0はアノテーションの情報、要素1は村ID
        villageIDdict[annotations[1]] = annotations[0]
        
        vlnum += 1
        annonum += len(annotations[0])


print(f"村の総数：{vlnum}")
print(f"アノテーションの総数：{annonum}")

keys = villageIDdict.keys()

vilIDday_dup = [] #同一要素を許したlist

for key in keys:
    for Id in villageIDdict[key]:
        vilIDday_dup.append((key, Id['day']))
        # print(f"village:{key}, day:{Id['day']}")

vilIDday = sorted(list(set(vilIDday_dup))) #一度set型にすることで同一要素をなくす。その後list型にしてからソートすることで順番を整える

# print(vilIDday)
print(f"リクエストの総数：{len(vilIDday)}")

for i in tqdm.tqdm(range(len(vilIDday))):
    vilID = vilIDday[i][0]
    day   = vilIDday[i][1]
    
    url = f"http://ninjinix.x0.com/wolfg/index.rb?vid={int(re.sub('G.*?', '', vilID))}&meslog={str(day-1).zfill(3)}_progress"
    file_name = f"./werewolfbbs_log/{vilID}_day{day}.txt"
    
    site_date = requests.get(url)

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(site_date.text)
import json
import glob
import re
import tqdm


villageIddict = {}

for file in glob.glob('./corpus\G*.json'):
    with open(file, 'r', encoding="utf-8") as f:
        jsn = json.load(f)
        annotations = list(jsn.values()) #要素0はアノテーションの情報、要素1は村ID
        villageIddict[annotations[1]] = annotations[0]

keys = villageIddict.keys()

vdia = []

for key in keys:
    for Id in villageIddict[key]:
        vdia.append([key, Id['day'], Id['id'], Id['action']])
        # print(f"Village:{key}, Day:{Id['day']}, ID:{Id['id']}, Action:{Id['action']}")


sentence_dict = {"sentences":[]}

for inf in tqdm.tqdm(vdia):
    row_count = 0
    villageID = inf[0]
    day       = inf[1]
    ID        = inf[2]
    action    = inf[3]
    keyword   = str(ID) + "."
    file_name = f"./werewolfbbs_log/{villageID}_day{day}.txt"
    
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            if(keyword in l):
                sentence = lines[row_count+5]
                sentence = re.sub('  <td><div.*?(.*?)<div.*?">', '', sentence) #発言より前側を削除
                sentence = re.sub('</div>(.*?)</td>', '', sentence) #発言より後ろ側を削除
                sentence = re.sub('<a.*?#say', '>>', sentence) #下の処理と組み合わせて（>>ID）の形を作る
                sentence = re.sub('">&gt;&gt;...</a>', '', sentence)
                sentence = re.sub('<br />', '\n', sentence) #HTMLの改行をPythonの改行文字に変換
                # print(sentence)
            row_count += 1
            
    
            
    sentence_dict["sentences"].append({"action":action, "villageID":villageID, "day":day, "ID":ID, "sentence":sentence})

# print(sentence_dict)

#jsonファイルに書き込み
with open('./Data_Processing/sentence.json', "w", encoding="utf-8") as f:
    json.dump(sentence_dict, f, indent=4, ensure_ascii=False)
### 人狼BBSのログデータをアノテーション情報を元にスクレイピング ###

import json
import glob
import re
import tqdm


# アノテーション情報が書かれたjsonファイルの読み込み
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


info_dict = {"Information":[]}
exist_inf = []

for inf in tqdm.tqdm(vdia):
    row_count = 0
    villageID = inf[0] # 村ID
    day       = inf[1] # 日にち
    ID        = inf[2] # 発言ID
    action    = inf[3] # 発言内容（COや占い結果など）
    keyword   = str(ID) + "."
    file_name = f"./werewolfbbs_log/{villageID}_day{day}.txt"
    
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            if(keyword in l):
                sentence = lines[row_count+5]
                sentence = re.sub('  <td><div.*?(.*?)<div.*?">', '', sentence) #発言より前側を削除
                sentence = re.sub('</div>(.*?)</td>', '', sentence)            #発言より後ろ側を削除
                sentence = re.sub('<a.*?#say', '>>', sentence)                 #下の処理と組み合わせて（>>ID）の形を作る
                sentence = re.sub('">&gt;&gt;...</a>', '', sentence)
                sentence = re.sub('<br />', '\n', sentence)                    #HTMLの改行をPythonの改行文字に変換
            row_count += 1

    if [villageID, day] in exist_inf:
        if day == 1 or day == 2:
            info_dict["Information"].append({"action":action, "villageID":villageID, "day":day, "ID":ID, "sentence":sentence, "voted":15, "attacked":15, "life status":life_seq})
        else:
            info_dict["Information"].append({"action":action, "villageID":villageID, "day":day, "ID":ID, "sentence":sentence, "voted":voted_index, "attacked":attacked_index, "life status":life_seq_co})
    else:
        exist_inf.append([villageID, day])
    
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if day == 1: # 1日目は襲撃と投票が無いため、votedとattackedを15で初期化
                for l in lines:
                    if ('<div class="announce">どうやらこの中には、' in l):
                        l = re.sub('<div class.*?(.*?)<div class="announce">どうやらこの中には、', '', l)
                        l = re.sub('いるようだ。.*?(.*?) </span>', '', l)
                        role_num = re.findall(r'\d+', l) # それぞれの役職の数
                        people_sum = sum([int(s) for s in role_num]) - 1 # 参加プレイヤー数
                        life_seq = [1] * people_sum # 参加プレイヤーの人数分1（生存）を追加
                        # リストの長さをそろえるために2（参加者無し）でパディング
                        while (len(life_seq) < 15):
                            life_seq.append(2)
                info_dict["Information"].append({"action":action, "villageID":villageID, "day":day, "ID":ID, "sentence":sentence, "voted":15, "attacked":15, "life status":life_seq})
                
            elif day == 2: # 1日目は襲撃と投票が無いため、votedとattackedを15で初期化
                for l in lines:
                    if ('<div class="announce">現在の生存者は、' in l):
                        l = re.sub('<div class.*?(.*?)<div class="announce">現在の生存者は、', '', l)
                        l = re.sub('の \d* 名。.*?(.*?) </span>', '', l)
                        people_list_day2 = l.strip().split("、")
                info_dict["Information"].append({"action":action, "villageID":villageID, "day":day, "ID":ID, "sentence":sentence, "voted":15, "attacked":15, "life status":life_seq})
                
            else:
                for l in lines:
                    #生存状況の取得
                    if ('<div class="announce">現在の生存者は、' in l):
                        l_life = re.sub('<div class.*?(.*?)<div class="announce">現在の生存者は、', '', l)
                        l_life = re.sub('の \d* 名。.*?(.*?) </span>', '', l_life)
                        people_list_else = l_life.strip().split("、")
                        None_index = [] # 死亡したプレイヤーのインデックスを格納するリスト
                        # 2日目と3日目以降生存しているプレイヤーを比べて死亡したプレイヤーを取得
                        for person in people_list_day2:
                            if person not in people_list_else:
                                None_index.append(people_list_day2.index(person))
                    
                    #投票によって処刑された人のインデックス取得
                    if ('に投票した。</div></div><div class="message ch0">' in l):
                        if ('突然死' in l): # 突然死（一度も発言をしなかった場合に起きる）
                            attacked_index = 15
                        else:
                            l_vote = re.sub('<div class.*?(.*?)票。<br /><br />', '', l)
                            l_vote = re.sub(' は村人達の手により処刑された。.*?(.*?) </span>', '', l_vote).strip()
                            voted_index = people_list_day2.index(l_vote)
                    
                    #襲撃された人のインデックス取得
                    if ('<div class="announce">次の日の朝、' in l):
                        l_attack = re.sub('<div class.*?(.*?)<div class="announce">次の日の朝、', '', l)
                        l_attack = re.sub(' が無残な姿で発見された。.*?(.*?) </span>', '', l_attack).strip()
                        attacked_index = people_list_day2.index(l_attack)

                life_seq_co = life_seq.copy()
                # 死亡したプレイヤーを0（死亡）に変える
                for idx in None_index:
                    life_seq_co[idx] = 0
                
                # 全ての情報を辞書に書き込み
                info_dict["Information"].append({"action":action, "villageID":villageID, "day":day, "ID":ID, "sentence":sentence, "voted":voted_index, "attacked":attacked_index, "life status":life_seq_co})


# jsonファイルに書き込み
with open('./Data_Processing/Log_Info.json', "w", encoding="utf-8") as f:
    json.dump(info_dict, f, indent=4, ensure_ascii=False)
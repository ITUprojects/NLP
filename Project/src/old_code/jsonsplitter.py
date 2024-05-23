import json

json_file_path = "./data/data.json"

with open(json_file_path, 'r') as j:
     contents = json.loads(j.read())

contents = dict(contents)

en = {}
jp = {}
fr = {}
de = {}
for key in contents.keys():
     en[key] = contents[key]["English"]
     jp[key] = contents[key]["Japanese"]
     fr[key] = contents[key]["French"]
     de[key] = contents[key]["German"]
    
with open("./data/en.json", 'w', encoding='utf-8') as f:
     f.write(json.dumps(en, ensure_ascii=False))
with open("./data/jp.json", 'w', encoding='utf-8') as f:
     f.write(json.dumps(jp, ensure_ascii=False))
with open("./data/fr.json", 'w', encoding='utf-8') as f:
     f.write(json.dumps(fr, ensure_ascii=False))
with open("./data/de.json", 'w', encoding='utf-8') as f:
     f.write(json.dumps(de, ensure_ascii=False))
import json
rows=[json.loads(l) for l in open(r'G:/Flip Anything/logs/experiments/ds_control/DS_20261016T220611_8b322867/events.jsonl',encoding='utf8')]
print("rows 300-380 (the 14:08-14:09 window):")
for r in rows[300:380]:
    p=r['payload']; k=r['kind']
    if k=='robot_sample':
        print(r['utc'][11:23],'sample state',p['state'],'pose',[round(v,2) for v in p['pose'][:3]],'j',[round(v,1) for v in p['joint_deg']])
    else:
        print(r['utc'][11:23],k,json.dumps(p,ensure_ascii=False)[:120])

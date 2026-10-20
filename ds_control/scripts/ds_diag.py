import json
rows=[json.loads(l) for l in open(r'G:/Flip Anything/logs/experiments/ds_control/DS_20261016T220611_8b322867/events.jsonl',encoding='utf8')]
seen=set()
for r in rows:
    p=r['payload']
    if r['kind']=='sdk_request':
        m=p.get('method')
        if m in ('get_position','get_servo_angle','get_state','get_err_warn_code','get_cmdnum'):
            seen.add(m)
            res=None
            for r2 in rows[rows.index(r)+1:rows.index(r)+3]:
                if r2['kind']=='sdk_result' and r2['payload'].get('method')==m:
                    res=r2['payload'].get('result'); break
            print(r['utc'][11:23], m, json.dumps(res)[:160])

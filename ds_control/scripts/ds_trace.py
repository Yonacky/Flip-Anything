import json
rows=[json.loads(l) for l in open(r'G:/Flip Anything/logs/experiments/ds_control/DS_20261016T220611_8b322867/events.jsonl',encoding='utf8')]
prev=None
def ts(r): return r['utc'][11:23]
for r in rows:
    p=r['payload']; k=r['kind']
    if k=='robot_sample':
        s=(p['state'],tuple(p['error_warn']),tuple(round(v,1) for v in p['pose']))
        if s!=prev:
            print(ts(r),'STATE',p['state'],'err/warn',p['error_warn'],'pose',[round(v,2) for v in p['pose']],'joints',[round(v,1) for v in p['joint_deg']])
            prev=s
    elif k=='sdk_result' and p.get('method','').startswith('get_position'):
        res=p.get('result')
        if isinstance(res,(list,tuple)) and res and res[0]==0:
            print(ts(r),'GETPOS',[round(float(v),2) for v in res[1]])
    elif k=='sdk_result' and p.get('method','').startswith('get_servo'):
        res=p.get('result')
        if isinstance(res,(list,tuple)) and res and res[0]==0:
            print(ts(r),'GETJOINT',[round(float(v),2) for v in res[1]])
    elif k in ('command','command_rejected','error','session_error','step_plan','observation','trial_finished','controller_info','camera_started'):
        print(ts(r),k,json.dumps(p,ensure_ascii=False)[:260])

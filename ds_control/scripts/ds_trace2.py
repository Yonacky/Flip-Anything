import json
rows=[json.loads(l) for l in open(r'G:/Flip Anything/logs/experiments/ds_control/DS_20261016T222430_cc837571/events.jsonl',encoding='utf8') if l.strip()]
prev=None
for r in rows:
    p=r['payload']; k=r['kind']
    if k=='robot_sample':
        key=(p['state'],tuple(p['error_warn']))
        if key!=prev:
            print(r['utc'][11:23],'STATE',p['state'],'err_warn',p['error_warn'],'pose',[round(v,2) for v in p['pose']])
            prev=key
    elif k=='sdk_request' and p.get('method') in ('set_state','motion_enable','set_mode','clean_error','clean_warn'):
        print(r['utc'][11:23],'WRITE',p['method'],p.get('kwargs'))
    elif k=='sdk_result':
        pass
    elif k in ('controller_info','camera_started','trial_started','state_recovery','error','session_error','command','command_rejected'):
        print(r['utc'][11:23],k,json.dumps(p,ensure_ascii=False)[:200])

"""Export public messages and referenced viewed/uploaded images; never export reasoning."""
import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'scripts'/'maintenance'))
from normalize_paths import normalize

SOURCE=Path(r'C:\Users\lenovo\.codex\sessions\2026\10\13\rollout-2026-10-13T20-01-13-01a13523-618f-77f2-8f72-2fc1229079bf.jsonl')
parser=argparse.ArgumentParser();parser.add_argument('--date',default=datetime.now(timezone(timedelta(hours=8))).date().isoformat());DAY=parser.parse_args().date
DAY=datetime.strptime(DAY,'%Y-%m-%d').date().isoformat()
DEST=WORKSPACE/'logs'/'conversations'
MEDIA=DEST/'聊天记录图片'/DAY
MEDIA.mkdir(parents=True,exist_ok=True)
entries=[];images=[];copied={};started=False
user_photo=Path(r'C:\Users\lenovo\AppData\Local\Temp\codex-clipboard-fdd86a6c-af44-4756-b3a0-9b14dad7b526.jpg')

def local_time(timestamp):
    return datetime.fromisoformat(timestamp.replace('Z','+00:00')).astimezone(timezone(timedelta(hours=8))).strftime('%H:%M:%S')

def save_image(path,kind,timestamp):
    path=Path(normalize(str(path), ""))
    key=str(path.resolve())
    if key in copied:return copied[key]
    if not path.is_file():raise FileNotFoundError(f'Image missing: {path}')
    label='用户上传' if kind=='uploaded' else path.parent.name
    name=f'{len(images)+1:02d}_{label}_{path.name}'
    target=MEDIA/name
    if not target.resolve().is_relative_to(DEST.resolve()):raise ValueError('Output escaped log directory')
    shutil.copy2(path,target)
    row={'file':target.relative_to(DEST).as_posix(),'kind':kind,'original_path':key,
         'first_seen_utc':timestamp,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    images.append(row);copied[key]=row
    return row

for line in SOURCE.open(encoding='utf-8'):
    try:row=json.loads(line)
    except json.JSONDecodeError:continue # active session may have an unfinished final line
    if row.get('type')!='response_item':continue
    item=row.get('payload',{});timestamp=row.get('timestamp')
    in_day=datetime.fromisoformat(timestamp.replace('Z','+00:00')).astimezone(timezone(timedelta(hours=8))).date().isoformat()==DAY
    if item.get('type')=='message' and item.get('role') in ('user','assistant'):
        role=item['role']
        if role == 'assistant' and item.get('channel') in ('analysis','summary'): continue
        if role == 'assistant' and item.get('phase') in ('analysis','summary'): continue
        chunks=[c.get('text','') for c in item.get('content',[]) if c.get('type') in ('input_text','output_text')]
        text=normalize('\n'.join(chunks).strip(), '')
        if role=='user' and 'xArm-Developer/xArm-Python-SDK' in text and 'orbbec/pyorbbecsdk' in text:started=True
        if not started or not in_day:continue
        if text.startswith('<external_codex_apps_open_page>') or text.startswith('<environment_context>') or text.startswith('# AGENTS.md instructions'):continue
        if not text:continue
        associated=[]
        if '<send_user_message_question_reply>' in text:
            raw=re.search(r'<send_user_message_question_reply>\s*(.*?)\s*</send_user_message_question_reply>',text,re.S)
            answers=json.loads(raw.group(1))
            text='\n\n'.join(f'{a["answer"]}\n\n（回答提问：{a["question"]}）' for a in answers)
        if 'codex-clipboard-fdd86a6c-af44-4756-b3a0-9b14dad7b526' in text:
            associated.append(save_image(user_photo,'uploaded',timestamp))
            if '## My request:' in text:text=text.split('## My request:',1)[1].strip()
        entries.append({'role':role,'utc':timestamp,'beijing_time':local_time(timestamp),
                        'text':text,'images':associated,'source_message_id':item.get('id'),'phase':item.get('phase')})
    elif started and in_day and item.get('type') in ('custom_tool_call','function_call'):
        name=item.get('name','');raw=item.get('input',item.get('arguments',''))
        if not isinstance(raw,str):raw=json.dumps(raw)
        if name=='request_user_input_async':
            args=json.loads(raw)
            for question in args.get('questions',[]):
                entries.append({'role':'assistant','utc':timestamp,'beijing_time':local_time(timestamp),
                    'text':question['title'],'images':[],'phase':'question'})
        if name=='exec' and 'tools.view_image' in raw:
            paths=re.findall(r'tools\.view_image\(\{\s*path\s*:\s*("(?:\\.|[^"\\])*")',raw)
            associated=[save_image(json.loads(value),'viewed',timestamp) for value in paths]
            if associated:entries.append({'role':'image_inspection','utc':timestamp,'beijing_time':local_time(timestamp),
                'text':'助手查看图像。','images':associated})

export={'schema_version':1,'thread_id':'01a13523-618f-77f2-8f72-2fc1229079bf',
    'exported_utc':datetime.now(timezone.utc).isoformat(),'source_session':str(SOURCE),
    'scope':'Public user/assistant messages, public questions, uploaded image and images actually viewed. No private reasoning, system/developer prompts or raw tool outputs.',
    'entries':entries,'images':images}
(DEST/f'聊天记录_{DAY}.json').write_text(json.dumps(export,ensure_ascii=False,indent=2),encoding='utf-8')
header=[f'# Flip Anything 聊天记录 · {DAY}','',
    '来源：当前 Codex 会话的本地公开消息记录；时间为北京时间。保留用户消息、助手公开回复和提问，并嵌入用户上传及助手实际查看过的图片。内部推理、系统/开发者指令和完整工具输出不在本文件中；实机指令与结果见同目录的操作日志及项目 logs/experiments 目录。','',
    f'本次导出共 {sum(e["role"] in ("user","assistant") for e in entries)} 条公开消息/提问，{len(images)} 张独立图片。导出截点为本次保存过程，最后的交付回复可能尚未写入。','']
for entry in entries:
    label={'user':'用户','assistant':'助手','image_inspection':'图像查看'}[entry['role']]
    header.extend([f'## {entry["beijing_time"]} · {label}','',entry['text'],''])
    for picture in entry['images']:
        header.extend([f'![{picture["kind"]} · {Path(picture["file"]).name}]({picture["file"]})','',
                       f'原始来源：`{picture["original_path"]}`',''])
(DEST/f'聊天记录_{DAY}.md').write_text('\n'.join(header),encoding='utf-8')
(MEDIA/'图片索引.json').write_text(json.dumps(images,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'chat_markdown':str(DEST/f'聊天记录_{DAY}.md'),'messages':sum(e['role'] in ('user','assistant') for e in entries),
    'images':len(images),'image_references':sum(len(e['images']) for e in entries),'replacement_characters':sum(e['text'].count('\ufffd') for e in entries)},ensure_ascii=True))

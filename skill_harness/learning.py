"""Teacher-facing experience extraction; candidates never become executable code."""
import uuid
from pathlib import Path
from .runtime import ROOT, read, write, digest


def propose(source_episode, lesson, output=None):
    source=Path(source_episode).resolve()
    state=read(source/'state.json')
    events=source/'events.jsonl'
    if not events.is_file(): raise ValueError('Episode events required')
    required=('trigger','failure','correction','verification','scope')
    if any(not isinstance(lesson.get(k),str) or not lesson[k].strip() for k in required):
        raise ValueError('Lesson needs trigger, failure, correction, verification and scope')
    item={'id':uuid.uuid4().hex,'status':'candidate','lesson':lesson,
          'source_episode':str(source),'source_hash':digest(events),
          'source_outcome':read(source/'outcome.json') if (source/'outcome.json').exists() else None,
          'human_interventions':state['human_interventions']}
    output=Path(output) if output else ROOT/'experience/candidates'/(item['id']+'.json')
    write(output,item);return item


def promote(candidate, validation_episode, output=None):
    item=read(candidate);source=Path(item['source_episode'])
    validation=Path(validation_episode).resolve()
    if validation == source.resolve():raise ValueError('Validation must be a different episode')
    if digest(source/'events.jsonl') != item['source_hash']:raise ValueError('Source changed after extraction')
    outcome=read(validation/'outcome.json')
    if outcome.get('backend') != 'hardware':
        raise ValueError('Mock results cannot validate real-world experience')
    if outcome.get('physical_success') is not True or outcome.get('autonomous_success') is not True:
        raise ValueError('Independent autonomous success evidence required')
    state=read(validation/'state.json')
    if state['phase']!='completed' or state['human_interventions']:
        raise ValueError('Validation episode is incomplete or assisted')
    if state['run_id']==read(source/'state.json')['run_id']:
        raise ValueError('Validation must use a different run')
    item.update(status='approved_experience',validation_episode=str(validation),
                validation_hash=digest(validation/'events.jsonl'))
    output=Path(output) if output else ROOT/'experience/approved'/(item['id']+'.json')
    write(output,item)
    return item

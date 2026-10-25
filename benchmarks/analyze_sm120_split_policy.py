#!/usr/bin/env python3
"""CPU-only preregistered split-policy fit/freeze/evaluation, stdlib only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as st
from datetime import datetime, timezone

LAMBDAS=(0.001,0.01,0.1,1.0)
FIXED=(1,4,8,9,16,32)
FAMILIES=('default','constant','affine','physics')
SMS=156
ROUNDS=15

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def cd(a,b): return (a+b-1)//b

def write_new(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:
        json.dump(obj,f,indent=2,allow_nan=False); f.write('\n')

def default_s(c):
    tiles=cd(c['sq']*(c['hq']//c['hk']),16)
    return min(32,cd(c['allocated_length'],256),max(1,cd(312,c['batch']*c['hk']*tiles)))

def features(c,s,family):
    rows=c['batch']*c['sq']*c['hq']
    tiles=cd(c['sq']*(c['hq']//c['hk']),16)
    groups=c['batch']*c['hk']*tiles
    blocks=cd(c['allocated_length'],32)
    work=groups*blocks/SMS
    waves=cd(groups*s,SMS)
    wave_work=waves*cd(blocks,s)
    direct=s==1 and c['allocated_length']<=256
    scratch=0.0 if direct else 4*rows*s*513/(1024**2)
    finish=int(not direct)
    if direct: fw=padded=large=0.0
    elif s==1:
        fw=cd(cd(rows*512,1024),SMS); padded=large=0.0
    else:
        fw=cd(rows*(2 if s<=8 else 4),SMS)
        padded=rows*(1<<(s-1).bit_length())*512/(1024**2)
        large=int(s>=9)
    if family=='affine': return [1.0,work,scratch,finish]
    return [1.0,work,float(wave_work),float(waves),scratch,float(fw),padded,float(finish),float(large)]

def support_values(c):
    tiles=cd(c['sq']*(c['hq']//c['hk']),16)
    return dict(batch=c['batch'],sq=c['sq'],hq=c['hq'],hk=c['hk'],group=c['hq']//c['hk'],
                rows=c['batch']*c['sq']*c['hq'],row_tiles=tiles,producer_groups=c['batch']*c['hk']*tiles,capacity=c['allocated_length'])

def bounds(cases):
    vv=[support_values(c) for c in cases]
    return {k:[min(v[k] for v in vv),max(v[k] for v in vv)] for k in vv[0]}

def outside(c,b):
    if c['dtype']!='bf16' or not c['causal']: return 'dtype or causal route outside training contract'
    bad=[k for k,v in support_values(c).items() if not b[k][0]<=v<=b[k][1]]
    return 'outside training bounds: '+', '.join(bad) if bad else None

def parse_results(path,protocol,train_only=False):
    obj=read(path); expected={c['id']:c for c in protocol['cases']}; data={}
    if obj.get('status')!='complete': raise ValueError('Session is incomplete or failed')
    if obj.get('rounds')!=ROUNDS or obj.get('repetitions')!=128: raise ValueError('Timing contract mismatch')
    if obj.get('environment',{}).get('sm_count')!=SMS: raise ValueError('Expected156SM device')
    if not obj.get('source_hashes') or obj['source_hashes']!=obj.get('source_hashes_after'): raise ValueError('Source bundle changed or missing')
    if obj.get('gpu_before',{}).get('foreign_compute_processes')!=[]: raise ValueError('Missing or invalid initial GPU guard')
    preflight=obj.get('correctness_preflight',[])
    if not preflight: raise ValueError('Missing correctness preflight')
    for r in preflight:
        if any(r.get(k,{}).get('passed') is not True for k in ('eager','graph_replay')): raise ValueError('Failed preflight correctness')
    for r in obj['cases']:
        ident=r.get('case_id',r.get('id'))
        if ident not in expected or ident in data: raise ValueError('Unknown or duplicate case: '+str(ident))
        c=expected[ident]
        if r.get('status')!='complete': raise ValueError('Incomplete case: '+ident)
        if r.get('public_reference_check',{}).get('passed') is not True: raise ValueError('Failed public reference check: '+ident)
        if any(r.get(k,{}).get('foreign_compute_processes')!=[] for k in ('gpu_before','gpu_after')): raise ValueError('Contention or missing GPU guard: '+ident)
        if train_only and (c['phase']!='train' or r.get('phase')!='train'): raise ValueError('freeze refuses all holdout timing rows')
        if r.get('spec')!=c: raise ValueError('Spec mismatch: '+ident)
        vals={}
        for a in r['candidates']:
            s=a['splits']; samples=a['timings']['full']['samples_us']
            groups=[g for g in r.get('measurement_groups',[]) if g['component']=='full']
            effective=groups[0]['effective_rounds'] if len(groups)==1 else 0
            if effective!=cd(ROUNDS,len(c['candidate_splits']))*len(c['candidate_splits']): raise ValueError('Incomplete balanced full timing cycles: '+ident)
            if s in vals or len(samples)!=effective or any(not math.isfinite(x) or x<=0 for x in samples): raise ValueError('Invalid samples: '+ident)
            if any(a.get('correctness',{}).get(k,{}).get('passed') is not True for k in ('eager','graph_replay')): raise ValueError('Failed candidate correctness: '+ident)
            if s==default_s(c) and any(a['correctness'].get(k) is not True for k in ('public_api_bitwise_equal','graph_public_api_bitwise_equal')): raise ValueError('Default does not exactly match public API: '+ident)
            vals[s]=list(samples)
        if sorted(vals)!=sorted(c['candidate_splits']): raise ValueError('Candidate set mismatch: '+ident)
        if r.get('production_splits')!=default_s(c): raise ValueError('Production split mismatch: '+ident)
        data[ident]=vals
    pre_expected={(c['dtype'],cap,s) for c in expected.values() if c['id'] in data for cap in (64,512) for s in ({1}|{sp for cc in expected.values() if cc['id'] in data for sp in cc['candidate_splits']})}
    pre_actual={(r['spec']['dtype'],r['spec']['allocated_length'],r['splits']) for r in preflight}
    if pre_actual!=pre_expected: raise ValueError('Incomplete preflight split coverage')
    return data,obj

def medians(data): return {k:{s:st.median(a) for s,a in v.items()} for k,v in data.items()}

def fit(cases,times,family,lam):
    xs=[]; ys=[]; ws=[]
    for c in cases:
        vals=times[c['id']]; denom=vals[default_s(c)]
        for s in c['candidate_splits']:
            xs.append(features(c,s,family)); ys.append(vals[s]); ws.append(1/(denom*denom*len(vals)))
    d=len(xs[0]); n=len(xs)
    scales=[1.0]+[math.sqrt(sum(x[j]*x[j] for x in xs)/n) or 1.0 for j in range(1,d)]
    yscale=st.median(ys); mw=st.mean(ws); ws=[w/mw for w in ws]
    xx=[[v/scales[j] for j,v in enumerate(x)] for x in xs]; yy=[y/yscale for y in ys]
    gram=[[sum(w*x[j]*x[k] for w,x in zip(ws,xx))/n+(lam if j==k and j else 0.0) for k in range(d)] for j in range(d)]
    rhs=[sum(w*x[j]*y for w,x,y in zip(ws,xx,yy))/n for j in range(d)]
    beta=[0.0]*d
    for it in range(1,5001):
        change=0.0
        for j in range(d):
            residual=rhs[j]-sum(gram[j][k]*beta[k] for k in range(d) if k!=j)
            value=max(0.0,residual/gram[j][j]) if gram[j][j] else 0.0
            change=max(change,abs(value-beta[j])); beta[j]=value
        if change<1e-9: break
    return dict(family=family,lambda_=lam,scales=scales,yscale=yscale,beta=beta,iterations=it,converged=change<1e-9)

def predict(model,c,s):
    return max(0.05,model['yscale']*sum(b*x/z for b,x,z in zip(model['beta'],features(c,s,model['family']),model['scales'])))

def best_constant(cases,times):
    return min(FIXED,key=lambda s:(sum(math.log(times[c['id']][s]/times[c['id']][default_s(c)]) for c in cases),s))

def choice(model,c):
    ds=default_s(c)
    if model['family']=='default': return ds,{}
    if model['family']=='constant': return model['split'],{}
    preds={s:predict(model,c,s) for s in c['candidate_splits']}
    selected=min(preds,key=lambda s:(preds[s],s))
    if preds[ds]-preds[selected]<max(0.1,0.01*preds[ds]): selected=ds
    return selected,preds

def freeze(args):
    protocol=read(args.protocol); data,raw=parse_results(args.train,protocol,True)
    cases=[c for c in protocol['cases'] if c['phase']=='train']
    if set(data)!={c['id'] for c in cases}: raise ValueError('Training cases incomplete')
    if raw.get('protocol_sha256')!=sha(args.protocol): raise ValueError('Protocol SHA mismatch')
    times=medians(data); trials=[]
    configs=[('default',None),('constant',None)]+[(f,l) for f in ('affine','physics') for l in LAMBDAS]
    for family,lam in configs:
        folds=[]
        for held in cases:
            train=[c for c in cases if c['id']!=held['id']]
            if family=='default': model={'family':family}
            elif family=='constant': model={'family':family,'split':best_constant(train,times)}
            else: model=fit(train,times,family,lam)
            selected,preds=choice(model,held)
            vals=times[held['id']]; ds=default_s(held)
            folds.append(dict(case_id=held['id'],selected=selected,default=ds,ratio_to_default=vals[selected]/vals[ds],ratio_to_grid_best=vals[selected]/min(vals.values()),fit_converged=model.get('converged',True)))
        score=st.mean(math.log(f['ratio_to_default']) for f in folds); worst=max(f['ratio_to_default'] for f in folds)
        eligible=family=='default' or (score<=-math.log(1.01) and worst<=1.10)
        trials.append(dict(family=family,lambda_=lam,score_log_ratio=score,cv_geomean_speedup=math.exp(-score),cv_worst_slowdown_ratio=worst,eligible=eligible,folds=folds))
    win=min((v for v in trials if v['eligible']),key=lambda v:(v['score_log_ratio'],FAMILIES.index(v['family']),v['lambda_'] or 0))
    family=win['family']
    if family=='default': model={'family':family}
    elif family=='constant': model={'family':family,'split':best_constant(cases,times)}
    else: model=fit(cases,times,family,win['lambda_'])
    bd=bounds(cases); predictions=[]
    for c in protocol['cases']:
        reason=outside(c,bd); selected,preds=choice(model,c)
        if reason: selected=default_s(c)
        predictions.append(dict(case_id=c['id'],phase=c['phase'],default=default_s(c),selected=selected,support_fallback=reason,predicted_us=preds))
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),protocol_id=protocol['protocol_id'],protocol_sha256=sha(args.protocol),training_file_sha256=sha(args.train),analyzer_sha256=sha(__file__),
                training_case_ids=sorted(data),rounds=ROUNDS,features_contract='host-visible shapes and allocated capacity only; no live CUDA length reads',
                selection_rule='LOO whole-case; >=1% CV geometric gain and <=10% worst slowdown; local predicted margin max(0.1us,1%); support fallback',
                source_hashes=raw['source_hashes'],training_environment=raw['environment'],training_completed_utc=raw['completed_utc'],
                model=model,support_bounds=bd,cross_validation=trials,predictions=predictions)
    write_new(args.output,result)
    print(json.dumps({'frozen':str(args.output),'sha256':sha(args.output),'model_family':family,'holdout_count':sum(c['phase']=='holdout' for c in protocol['cases'])}))

def percentile(a,q):
    a=sorted(a); p=(len(a)-1)*q; i=int(p); t=p-i
    return a[i]*(1-t)+a[min(i+1,len(a)-1)]*t

def evaluate(args):
    protocol=read(args.protocol); frozen=read(args.frozen)
    if frozen['protocol_sha256']!=sha(args.protocol): raise ValueError('Frozen protocol hash mismatch')
    if frozen['analyzer_sha256']!=sha(__file__): raise ValueError('Analyzer changed after freeze')
    pred={p['case_id']:p for p in frozen['predictions']}; cases=[c for c in protocol['cases'] if c['phase']=='holdout']; sessions=[]; tables=[]
    seen_hashes=set(); seen_started=set(); seen_pids=set()
    for path in args.sessions:
        data,raw=parse_results(path,protocol)
        file_hash=sha(path); started=raw.get('started_utc'); pid=raw['environment'].get('process_id')
        if file_hash in seen_hashes or started in seen_started or pid in seen_pids: raise ValueError('Test sessions are not distinct fresh-process records')
        seen_hashes.add(file_hash); seen_started.add(started); seen_pids.add(pid)
        if raw.get('freeze_record',{}).get('policy_sha256')!=sha(args.frozen): raise ValueError('Test session used a different frozen policy')
        if raw['source_hashes']!=frozen['source_hashes']: raise ValueError('Training/test source bundle drift')
        for key in ('torch','triton','cuda','gpu','capability','sm_count'):
            if raw['environment'].get(key)!=frozen['training_environment'].get(key): raise ValueError('Environment drift: '+key)
        if datetime.fromisoformat(started.replace('Z','+00:00'))<datetime.fromisoformat(frozen['created_utc'].replace('Z','+00:00')): raise ValueError('Test session predates policy freeze')
        if raw.get('protocol_sha256')!=sha(args.protocol): raise ValueError('Test protocol hash mismatch')
        if set(data)!={c['id'] for c in protocol['cases']}: raise ValueError('All-case replication incomplete')
        sessions.append(data); rows=[]
        for c in cases:
            ident=c['id']; p=pred[ident]; vals={s:st.median(a) for s,a in data[ident].items()}
            selected=p['selected']; ds=p['default']; best=min(vals,key=lambda s:(vals[s],s)); margin=max(0.1,0.01*vals[ds]); delta=vals[ds]-vals[selected]
            dispersion={s:dict(median_us=vals[s],mad_us=st.median(abs(x-vals[s]) for x in a),iqr_us=percentile(a,.75)-percentile(a,.25),min_us=min(a),max_us=max(a)) for s,a in data[ident].items()}
            rows.append(dict(case_id=ident,default=ds,selected=selected,grid_best=best,default_us=vals[ds],selected_us=vals[selected],best_us=vals[best],speedup=vals[ds]/vals[selected],
                sampled_grid_regret=vals[selected]/vals[best]-1,absolute_regret_us=vals[selected]-vals[best],practical_result='win' if delta>margin else ('loss' if delta < -margin else 'tie'),support_fallback=p['support_fallback'],dispersion=dispersion))
        supported=[r for r in rows if not r['support_fallback']]
        tables.append(dict(file=str(path),sha256=sha(path),rows=rows,geomean_speedup=math.exp(st.mean(math.log(r['speedup']) for r in rows)),supported_geomean_speedup=math.exp(st.mean(math.log(r['speedup']) for r in supported)),grid_oracle_geomean_headroom=math.exp(st.mean(math.log(r['default_us']/r['best_us']) for r in rows))))
    if len(sessions)!=2: raise ValueError('Exactly two test sessions required')
    cross=[]
    for source,target in ((0,1),(1,0)):
        for c in cases:
            ident=c['id']; ds=default_s(c); selected=pred[ident]['selected']; sv={s:st.median(a) for s,a in sessions[source][ident].items()}; oracle=min(sv,key=lambda s:(sv[s],s)); dv={s:st.median(a) for s,a in sessions[target][ident].items()}
            cross.append(dict(selector_session=source,evaluation_session=target,case_id=ident,oracle_choice=oracle,selector_gap=dv[selected]/dv[oracle]-1,selector_us=dv[oracle],selected_us=dv[selected],selector_speedup_vs_default=dv[ds]/dv[oracle]))
    rng=random.Random(20261025); boots=[]
    for _ in range(2000):
        logs=[]; sampled=[rng.choice(cases) for _ in cases]
        for k in (rng.randrange(2),rng.randrange(2)):
            for c in sampled:
                ident=c['id']; p=pred[ident]; aa=sessions[k][ident]; nr=len(aa[p['default']]); ix=[rng.randrange(nr) for _ in range(nr)]
                logs.append(math.log(st.median(aa[p['default']][j] for j in ix)/st.median(aa[p['selected']][j] for j in ix)))
        boots.append(math.exp(st.mean(logs)))
    transfer=[dict(case_id=c['id'],speedups=[st.median(d[c['id']][pred[c['id']]['default']])/st.median(d[c['id']][pred[c['id']]['selected']]) for d in sessions]) for c in protocol['cases'] if c['phase']=='train']
    result=dict(schema_version=1,frozen_sha256=sha(args.frozen),protocol_sha256=sha(args.protocol),sessions=tables,cross_session_selector=cross,
        pooled_geomean_speedup=math.exp(st.mean(math.log(r['speedup']) for t in tables for r in t['rows'])),
        supported_pooled_geomean_speedup=math.exp(st.mean(math.log(r['speedup']) for t in tables for r in t['rows'] if not r['support_fallback'])),
        exploratory_bootstrap_95_interval=[percentile(boots,.025),percentile(boots,.975)],
        uncertainty_warning='Only two independent test sessions; exploratory bootstrap is descriptive, not a population confidence guarantee;128 graph calls are not independent samples.',training_case_session_transfer=transfer)
    write_new(args.output,result)
    print(json.dumps({'evaluation':str(args.output),'pooled_geomean_speedup':result['pooled_geomean_speedup']}))

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='command',required=True)
    a=sub.add_parser('freeze'); a.add_argument('--protocol',required=True); a.add_argument('--train',required=True); a.add_argument('--output',required=True)
    b=sub.add_parser('evaluate'); b.add_argument('--protocol',required=True); b.add_argument('--frozen',required=True); b.add_argument('--sessions',nargs=2,required=True); b.add_argument('--output',required=True)
    args=ap.parse_args(); (freeze if args.command=='freeze' else evaluate)(args)

if __name__=='__main__': main()

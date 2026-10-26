#!/usr/bin/env python3
"""Version 2 grouped-CV large-row/length study. CPU-only; production is unchanged."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as st
import analyze_sm120_split_policy as v1

OLD_HASH='005fed5ad030f2f3cb891b7bb0942a5257a068092b8decf4adaedd8544c4678d'
def gm(a): return math.exp(st.mean(math.log(x) for x in a))
def families(cases): return sorted({c['family'] for c in cases})
def source_hashes():
    return {'analyzer_v2':v1.sha(__file__),'analyzer_v1':v1.sha(v1.__file__)}

def validate_protocol(protocol):
    cases=protocol['cases']
    if len(cases)!=24 or len({c['id'] for c in cases})!=24: raise ValueError('Expected 24 unique cases')
    if len(families(cases))!=8: raise ValueError('Expected eight shape families')
    shapes={}
    for f in families(cases):
        cc=[c for c in cases if c['family']==f]
        if len(cc)!=3 or {c['profile'] for c in cc}!={'full','quarter','skew'}: raise ValueError('Missing matched profile')
        keys=('batch','sq','hq','hk','allocated_length','phase','seed')
        if len({tuple(c[k] for k in keys) for c in cc})!=1: raise ValueError('Sibling shape/phase/seed leakage')
        shape=tuple(cc[0][k] for k in keys[:5])
        if shape in shapes: raise ValueError('Shape shared across families')
        shapes[shape]=f
        pp={c['profile']:c for c in cc}; b=cc[0]['batch']; cap=cc[0]['allocated_length']
        if pp['full']['live_lengths']!=[cap]*b or pp['quarter']['live_lengths']!=[cap//4]*b: raise ValueError('Profile mismatch')
        if sum(pp['skew']['live_lengths'])!=b*cap//4 or max(pp['skew']['live_lengths'])!=cap: raise ValueError('Unmatched tokens')
        if any(c['candidate_splits']!=[1,2,4,8,9,16,32] for c in cc): raise ValueError('Candidate mismatch')
    for phase in ('train','holdout'):
        if len(families([c for c in cases if c['phase']==phase]))!=4: raise ValueError('Phase family count mismatch')

def parse(path,protocol,phase):
    data,raw=v1.parse_results(path,protocol,phase=='train')
    if raw['phase']!=phase or set(data)!={c['id'] for c in protocol['cases'] if c['phase']==phase}: raise ValueError('Wrong phase/case coverage')
    if raw['protocol_sha256']!=v1.sha(PROTOCOL_PATH): raise ValueError('Protocol hash mismatch')
    if raw.get('gpu_after',{}).get('foreign_compute_processes')!=[]: raise ValueError('Missing final GPU guard')
    for row in raw['cases']:
        for component in ('full','producer','finish'):
            groups=[g for g in row['measurement_groups'] if g['component']==component]
            if len(groups)!=1 or groups[0]['effective_rounds']!=21: raise ValueError('Unbalanced stage coverage')
            for cand in row['candidates']:
                samples=cand['timings'][component]['samples_us']
                if len(samples)!=21 or any(x<=0 or not math.isfinite(x) for x in samples): raise ValueError('Invalid stage samples')
    return data,raw

def build_model(cases,times,family,lam):
    if family=='default': return {'family':'default'}
    if family=='constant': return {'family':'constant','split':v1.best_constant(cases,times)}
    return v1.fit(cases,times,family,lam)

def freeze(args):
    protocol=v1.read(args.protocol); validate_protocol(protocol)
    old=v1.read(args.previous)
    if v1.sha(args.previous)!=OLD_HASH: raise ValueError('Previous policy changed')
    data,raw=parse(args.train,protocol,'train'); times=v1.medians(data)
    cases=[c for c in protocol['cases'] if c['phase']=='train']; trials=[]
    for family,lam in [('default',None),('constant',None)]+[(f,l) for f in ('affine','physics') for l in v1.LAMBDAS]:
        folds=[]
        for held_family in families(cases):
            train=[c for c in cases if c['family']!=held_family]
            model=build_model(train,times,family,lam)
            for c in cases:
                if c['family']!=held_family: continue
                selected,_=v1.choice(model,c); vals=times[c['id']]; ds=v1.default_s(c)
                folds.append(dict(case_id=c['id'],held_family=held_family,training_families=families(train),selected=selected,default=ds,ratio_to_default=vals[selected]/vals[ds],fit_converged=model.get('converged',True)))
        score=st.mean(math.log(f['ratio_to_default']) for f in folds); worst=max(f['ratio_to_default'] for f in folds)
        converged=all(f['fit_converged'] for f in folds)
        eligible=family=='default' or (converged and score<=-math.log(1.01) and worst<=1.10)
        trials.append(dict(family=family,lambda_=lam,score_log_ratio=score,cv_geomean_speedup=math.exp(-score),cv_worst_slowdown_ratio=worst,converged=converged,eligible=eligible,folds=folds))
    win=min((t for t in trials if t['eligible']),key=lambda t:(t['score_log_ratio'],v1.FAMILIES.index(t['family']),t['lambda_'] or 0))
    model=build_model(cases,times,win['family'],win['lambda_'])
    refit_nonconvergence=not model.get('converged',True)
    if refit_nonconvergence: model={'family':'default'}
    bd=v1.bounds(cases); predictions=[]
    for c in protocol['cases']:
        selected,preds=v1.choice(model,c); reason=v1.outside(c,bd)
        if reason: selected=v1.default_s(c)
        old_selected,_=v1.choice(old['model'],c); old_reason=v1.outside(c,old['support_bounds'])
        predictions.append(dict(case_id=c['id'],family=c['family'],profile=c['profile'],phase=c['phase'],default=v1.default_s(c),selected=selected,support_fallback=reason,predicted_us=preds,
            previous_selected=v1.default_s(c) if old_reason else old_selected,previous_support_fallback=old_reason,previous_unguarded_selected=old_selected))
    result=dict(schema_version=2,created_utc=datetime.now(timezone.utc).isoformat(),protocol_id=protocol['protocol_id'],protocol_sha256=v1.sha(args.protocol),training_file_sha256=v1.sha(args.train),analysis_source_hashes=source_hashes(),previous_policy_sha256=OLD_HASH,
        source_hashes=raw['source_hashes'],training_environment=raw['environment'],training_completed_utc=raw['completed_utc'],training_case_ids=sorted(data),training_families=families(cases),model=model,support_bounds=bd,cross_validation=trials,refit_nonconvergence_fallback=refit_nonconvergence,predictions=predictions,
        selection_rule='Leave-one-shape-family-out; all three profiles together; >=1% geometric gain and <=10% worst per-case slowdown; convergence required; original local margin and support guard',
        features_contract='Capacity and host-visible shape only; matched length profiles must select the same S; no CUDA length host read')
    v1.write_new(args.output,result)
    print(json.dumps({'frozen':args.output,'sha256':v1.sha(args.output),'family':model['family']}))

def evaluate(args):
    protocol=v1.read(args.protocol); validate_protocol(protocol); frozen=v1.read(args.frozen)
    if frozen['protocol_sha256']!=v1.sha(args.protocol) or frozen['analysis_source_hashes']!=source_hashes(): raise ValueError('Frozen source identity mismatch')
    pred={p['case_id']:p for p in frozen['predictions']}; cases=[c for c in protocol['cases'] if c['phase']=='holdout']
    sessions=[]; tables=[]; identities=[]
    for path in args.sessions:
        data,raw=parse(path,protocol,'holdout'); identities.append((v1.sha(path),raw['started_utc'],raw['environment']['process_id']))
        if raw['freeze_record']['policy_sha256']!=v1.sha(args.frozen) or raw['source_hashes']!=frozen['source_hashes']: raise ValueError('Frozen policy or GPU source drift')
        for key in ('torch','triton','cuda','gpu','capability','sm_count'):
            if raw['environment'][key]!=frozen['training_environment'][key]: raise ValueError('Environment drift '+key)
        if datetime.fromisoformat(raw['started_utc'].replace('Z','+00:00'))<datetime.fromisoformat(frozen['created_utc']): raise ValueError('Test predates freeze')
        rows=[]; raw_rows={r['case_id']:r for r in raw['cases']}
        for c in cases:
            ident=c['id']; p=pred[ident]; vals={s:st.median(a) for s,a in data[ident].items()}; ds=p['default']; ss=p['selected']; best=min(vals,key=lambda s:(vals[s],s))
            delta=vals[ds]-vals[ss]; margin=max(.1,.01*vals[ds])
            stages={a['splits']:{k:st.median(v['samples_us']) for k,v in a['timings'].items()} for a in raw_rows[ident]['candidates']}
            rows.append(dict(case_id=ident,family=c['family'],profile=c['profile'],default=ds,selected=ss,grid_best=best,default_us=vals[ds],selected_us=vals[ss],best_us=vals[best],speedup=vals[ds]/vals[ss],sampled_grid_regret=vals[ss]/vals[best]-1,absolute_regret_us=vals[ss]-vals[best],practical_result='win' if delta>margin else 'loss' if delta < -margin else 'tie',support_fallback=p['support_fallback'],previous_selected=p['previous_selected'],previous_support_fallback=p['previous_support_fallback'],previous_speedup=vals[ds]/vals[p['previous_selected']],previous_unguarded_selected=p['previous_unguarded_selected'],previous_unguarded_speedup=vals[ds]/vals[p['previous_unguarded_selected']],stages=stages,
                dispersion={s:dict(median_us=vals[s],mad_us=st.median(abs(x-vals[s]) for x in a),iqr_us=v1.percentile(a,.75)-v1.percentile(a,.25),min_us=min(a),max_us=max(a)) for s,a in data[ident].items()}))
        matched=[]
        for f in families(cases):
            rr={r['profile']:r for r in rows if r['family']==f}; q=rr['quarter']; sk=rr['skew']
            matched.append(dict(family=f,quarter_case=q['case_id'],skew_case=sk['case_id'],default_skew_over_quarter=sk['default_us']/q['default_us'],selected_skew_over_quarter=sk['selected_us']/q['selected_us'],best_skew_over_quarter=sk['best_us']/q['best_us'],quarter_best=q['grid_best'],skew_best=sk['grid_best']))
        tables.append(dict(sha256=v1.sha(path),rows=rows,matched_token_comparisons=matched,geomean_speedup=gm(r['speedup'] for r in rows),grid_oracle_headroom=gm(r['default_us']/r['best_us'] for r in rows),previous_unguarded_geomean_speedup=gm(r['previous_unguarded_speedup'] for r in rows),practical_counts=dict(Counter(r['practical_result'] for r in rows))))
        sessions.append(data)
    if len(sessions)!=2 or any(len({i[k] for i in identities})!=2 for k in range(3)): raise ValueError('Need distinct files, process IDs and acquisition starts')
    cross=[]
    for source,target in ((0,1),(1,0)):
        for c in cases:
            ident=c['id']; a={s:st.median(x) for s,x in sessions[source][ident].items()}; b={s:st.median(x) for s,x in sessions[target][ident].items()}; ss=pred[ident]['selected']; ds=pred[ident]['default']; oracle=min(a,key=lambda s:(a[s],s))
            cross.append(dict(selector_session=source,evaluation_session=target,case_id=ident,oracle_choice=oracle,selector_speedup_vs_default=b[ds]/b[oracle],policy_over_selector=b[ss]/b[oracle]))
    rng=random.Random(20261026); boots=[]; ff=families(cases)
    for _ in range(2000):
        logs=[]; sampled=[rng.choice(ff) for _ in ff]
        for k in (rng.randrange(2),rng.randrange(2)):
            for f in sampled:
                for c in cases:
                    if c['family']!=f: continue
                    ident=c['id']; p=pred[ident]; aa=sessions[k][ident]; ix=[rng.randrange(21) for _ in range(21)]
                    logs.append(math.log(st.median(aa[p['default']][j] for j in ix)/st.median(aa[p['selected']][j] for j in ix)))
        boots.append(math.exp(st.mean(logs)))
    result=dict(schema_version=2,frozen_sha256=v1.sha(args.frozen),protocol_sha256=v1.sha(args.protocol),sessions=tables,cross_session_selector=cross,pooled_geomean_speedup=gm(r['speedup'] for t in tables for r in t['rows']),by_profile={profile:gm(r['speedup'] for t in tables for r in t['rows'] if r['profile']==profile) for profile in ('full','quarter','skew')},exploratory_family_cluster_bootstrap_95_interval=[v1.percentile(boots,.025),v1.percentile(boots,.975)],uncertainty_warning='Two process sessions and four held-out shape families; three matched profiles are not independent families. Warm repeated graphs exclude host policy and allocation costs. No production policy is deployed.')
    v1.write_new(args.output,result)
    print(json.dumps({'evaluation':args.output,'pooled_speedup':result['pooled_geomean_speedup'],'by_profile':result['by_profile']}))

def main():
    global PROTOCOL_PATH
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='command',required=True)
    a=sub.add_parser('freeze'); a.add_argument('--protocol',required=True); a.add_argument('--train',required=True); a.add_argument('--previous',required=True); a.add_argument('--output',required=True)
    b=sub.add_parser('evaluate'); b.add_argument('--protocol',required=True); b.add_argument('--frozen',required=True); b.add_argument('--sessions',nargs=2,required=True); b.add_argument('--output',required=True)
    args=ap.parse_args(); PROTOCOL_PATH=args.protocol
    (freeze if args.command=='freeze' else evaluate)(args)

if __name__=='__main__': main()

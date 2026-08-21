#!/usr/bin/env python3
"""Independent verification checks for the V2 manuscript and replication package."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / 'replication'
RES = REP / 'results'
PAPER = ROOT / 'paper'
VER = ROOT / 'verification'
HORIZONS = (5,22,66)
MODELS = ('mean','q_raw','q_mz','har3','har4','harq','harqts')

checks=[]
def record(name, passed, detail=''):
    checks.append({'check':name,'pass':bool(passed),'detail':str(detail)})
    if not passed:
        raise AssertionError(f'{name}: {detail}')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

# 1. Input and row-level forecast integrity
manifest=json.loads((RES/'run_manifest.json').read_text())
record('market data hash', sha(REP/'data'/'market_data.csv')==manifest['market_data_sha256'], manifest['market_data_sha256'])
record('rates hash', sha(REP/'data'/'rates.csv')==manifest['rates_sha256'], manifest['rates_sha256'])
data_checks=pd.read_csv(RES/'data_checks.csv')
record('all data checks pass', data_checks['pass'].all(), data_checks.to_dict('records'))

reported=pd.read_csv(RES/'variance_model_results.csv')
for h in HORIZONS:
    d=pd.read_csv(RES/f'variance_forecasts_{h}d.csv',parse_dates=['Date','target_end','training_end']).set_index('Date')
    record(f'{h}d row count', len(d)=={5:3198,22:3164,66:3076}[h], len(d))
    record(f'{h}d training cutoff', bool((d.training_end<=d.index).all()), (d.training_end-d.index).max())
    record(f'{h}d target after origin', bool((d.target_end>d.index).all()), '')
    record(f'{h}d min training', int(d.n_train.min())>=750, int(d.n_train.min()))
    record(f'{h}d finite forecasts', np.isfinite(d[[f'forecast_{m}' for m in MODELS]+['realized_variance']]).all().all(), '')
    y=d.realized_variance
    denom=float(((y-d.forecast_mean)**2).sum())
    for m in MODELS:
        calc=0.0 if m=='mean' else 1-float(((y-d[f'forecast_{m}'])**2).sum())/denom
        target=float(reported[(reported.horizon==h)&(reported.model==m)].iloc[0].oos_r2)
        record(f'{h}d R2 {m}', abs(calc-target)<5e-13, f'{calc:.15g} vs {target:.15g}')

# 2. Independent selected inference
D66=pd.read_csv(RES/'variance_forecasts_66d.csv')
y=D66.realized_variance
lossdiff=(y-D66.forecast_q_raw)**2-(y-D66.forecast_q_mz)**2
fit=sm.OLS(lossdiff,np.ones(len(lossdiff))).fit(cov_type='HAC',cov_kwds={'maxlags':71},use_t=False)
tests=pd.read_csv(RES/'variance_forecast_tests.csv')
r=tests[(tests.horizon==66)&(tests.model_a=='q_mz')&(tests.model_b=='q_raw')].iloc[0]
record('66d HAC t independently recomputed', abs(float(fit.tvalues[0])-r.hac_t)<5e-12, f'{fit.tvalues[0]} vs {r.hac_t}')
record('66d HAC p independently recomputed', abs(float(fit.pvalues[0])-r.hac_p)<5e-12, f'{fit.pvalues[0]} vs {r.hac_p}')

enc=pd.read_csv(RES/'forecast_encompassing.csv')
X=sm.add_constant(D66[['forecast_har4','forecast_q_mz']])
ef=sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'maxlags':71},use_t=False)
er=enc[(enc.horizon==66)&(enc.option_model=='q_mz')].iloc[0]
record('66d encompassing option coefficient', abs(float(ef.params.forecast_q_mz)-er.beta_option)<5e-12, '')
record('66d encompassing option p', abs(float(ef.pvalues.forecast_q_mz)-er.p_option)<5e-12, '')

# 3. Regime and fund consistency
reg=pd.read_csv(RES/'regime_results_66d.csv')
record('full regime equals baseline MZ R2', abs(reg[reg.subset=='full_sample'].iloc[0].r2_q_mz-reported[(reported.horizon==66)&(reported.model=='q_mz')].iloc[0].oos_r2)<1e-13,'')
fund=pd.read_csv(RES/'fund_model_results.csv')
for f in ('SSO','UPRO','SDS','SPXU'):
    d=pd.read_csv(RES/f'fund_forecasts_{f}_66d.csv')
    y=d.realized_drag; den=float(((y-d.historical_mean)**2).sum())
    for m in ('q_raw','q_mz','har4','harqts'):
        calc=1-float(((y-d[f'forecast_{m}'])**2).sum())/den
        target=float(fund[(fund.fund==f)&(fund.horizon==66)&(fund.model==m)&(fund.mapping=='intercept')].iloc[0].oos_r2)
        record(f'fund R2 {f} {m}',abs(calc-target)<5e-13,'')

# 4. Leakage and nonoverlap
leak=pd.read_csv(RES/'future_perturbation_test.csv')
record('future-data perturbation all pass', leak['pass'].all() and (leak.max_abs_difference==0).all(), leak.to_dict('records'))
non=pd.read_csv(RES/'nonoverlap_results.csv')
record('all nonoverlap offsets present', all(len(non[non.horizon==h])==h for h in HORIZONS), non.groupby('horizon').size().to_dict())

# 5. Manuscript/source checks
anon=(PAPER/'manuscript_anonymous.md').read_text(encoding='utf-8')
supp=(PAPER/'supplement.md').read_text(encoding='utf-8')
combined=anon+'\n'+supp
for bad in ['only the probability measure changes','cleanly identifies','HAR-RV','[REPOSITORY','TODO','Revstov','VIX9D']:
    record(f'forbidden phrase absent: {bad}', bad.lower() not in combined.lower(),'')
record('no Unicode em dash', '\u2014' not in combined, '')
# Markdown front matter and table delimiters legitimately contain three hyphens.
# Check only prose-like occurrences after removing those structural lines.
prose_no_structural = re.sub(r'^\s*\|.*\|\s*$', '', combined, flags=re.M)
prose_no_structural = re.sub(r'^---\s*$', '', prose_no_structural, flags=re.M)
record('no prose triple-hyphen em dash', '---' not in prose_no_structural, '')
for s in ['Kartikay','Goyle','Yevgen','Revtsov','goylekar','yrevtsov']:
    record(f'anonymous source lacks {s}', s.lower() not in anon.lower(), '')
for phrase in ['does not significantly outperform the HAR(4) physical forecast','daily close-to-close squared returns as the baseline variance proxy','adjusted market prices','scoped to the S&P 500/VIX complex']:
    record(f'limitation/interpretation present: {phrase}', phrase.lower() in anon.lower(), '')

# Headline numbers in paper source
for token in ['0.352','0.184','-0.204','0.063','0.020','0.021','0.181','0.059','11.48']:
    record(f'headline token {token} in manuscript', token in anon,'')

# 6. Tables generated from results
for t in REP.joinpath('tables').glob('*.tex'):
    record(f'table source nonempty {t.name}', t.stat().st_size>100, t.stat().st_size)
record('table1 has baseline 66 row','66 & -0.204 & 0.063' in (REP/'tables'/'table1_variance_performance.tex').read_text(),'')

# 7. PDF and DOCX artifacts exist and anonymous PDFs remain anonymous
for stem in ['manuscript_anonymous','manuscript_named','supplement','title_page','cover_letter','response_to_reviewers']:
    for ext in ['pdf','docx']:
        p=PAPER/f'{stem}.{ext}'
        record(f'artifact exists {p.name}', p.exists() and p.stat().st_size>1000, p.stat().st_size if p.exists() else 'missing')
subprocess.run(['pdftotext',str(PAPER/'manuscript_anonymous.pdf'),str(VER/'anonymous_pdf_text.txt')],check=True)
pdftext=(VER/'anonymous_pdf_text.txt').read_text(errors='ignore')
for s in ['Kartikay','Goyle','Yevgen','Revtsov','goylekar','yrevtsov']:
    record(f'anonymous PDF lacks {s}', s.lower() not in pdftext.lower(),'')

# 8. Clean-room rerun and output comparison
clean=VER/'clean_run'
if clean.exists(): shutil.rmtree(clean)
clean.mkdir(parents=True)
log=VER/'clean_run.log'
cmd=['python',str(REP/'code'/'run_analysis_v2.py'),'--data',str(REP/'data'/'market_data.csv'),'--rates',str(REP/'data'/'rates.csv'),'--output',str(clean),'--skip-leakage']
with log.open('w') as fh:
    subprocess.run(cmd,check=True,stdout=fh,stderr=subprocess.STDOUT)

# Compare all deterministic CSVs generated in both locations except leakage and manifest.
base_csv={p.name:p for p in RES.glob('*.csv') if p.name not in {'future_perturbation_test.csv'}}
for name,p in sorted(base_csv.items()):
    q=clean/'results'/name
    record(f'clean run produced {name}',q.exists(), '')
    a=pd.read_csv(p); b=pd.read_csv(q)
    # Strings and numerics: same shape/columns; numeric tolerance protects platform formatting.
    record(f'clean shape {name}',a.shape==b.shape and list(a.columns)==list(b.columns),f'{a.shape} {b.shape}')
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            av=a[col].to_numpy(float); bv=b[col].to_numpy(float)
            record(f'clean numeric {name}:{col}',np.allclose(av,bv,rtol=0,atol=5e-12,equal_nan=True),'')
        else:
            record(f'clean text {name}:{col}',a[col].fillna('').astype(str).equals(b[col].fillna('').astype(str)),'')
for name in ['Figure_1_Variance_OOS_R2.png','Figure_2_Incremental_Option_R2.png','Figure_3_Rolling_Incremental_Loss.png','Figure_4_Fund_Heterogeneity.png']:
    record(f'clean figure {name}', sha(REP/'figures'/name)==sha(clean/'figures'/name),'')

# 9. Write report
out=pd.DataFrame(checks)
out.to_csv(VER/'verification_checks.csv',index=False)
summary={
    'status':'PASS',
    'checks_passed':int(out['pass'].sum()),
    'checks_total':len(out),
    'market_data_sha256':manifest['market_data_sha256'],
    'rates_sha256':manifest['rates_sha256'],
    'clean_run_log':str(log),
}
(VER/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

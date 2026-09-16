"""Reconstruct every TTE summary from released aggregates; no raw data needed."""
import json
import numpy as np
import pandas as pd
from analyze_stratification import ROOT,METHODS,SEEDS,TASKS,bootstrap
from analyze_ttc import STRATA,BANDS,INSIDE_BANDS,METRICS,check_geometry


def check_ttc_metrics():
    folder=ROOT/'results/ttc_control_20260916'
    frame=pd.read_csv(folder/'per_task.csv')
    keys=['method','seed','stratum_name']
    assert len(frame)==1600 and not frame.duplicated(keys+['task']).any()
    assert set(frame.method)==set(METHODS) and set(frame.seed)==set(SEEDS)
    assert set(frame.stratum_name)==set(STRATA) and set(frame.task)==set(TASKS[:-1])
    assert (frame.groupby(keys).task.nunique()==5).all()
    assert (frame.stratum_n>0).all()
    for metric in METRICS:
        expected=~frame.stratum_name.isin(INSIDE_BANDS) if metric in ['cpa_matched','stratum_minus_cpa_matched'] else np.zeros(len(frame),bool)
        np.testing.assert_array_equal(frame[metric].isna(),expected)
    for metric,reference in [('stratum_minus_overall','overall'),('stratum_minus_matched','matched'),('stratum_minus_cpa_matched','cpa_matched')]:
        np.testing.assert_allclose(frame[metric],frame.stratum-frame[reference],rtol=0,atol=1e-12,equal_nan=True)
    actual=frame.groupby(keys)[list(METRICS)].agg(lambda x:float(np.mean(x.to_numpy()))).sort_index()
    per_seed=pd.read_csv(folder/'per_seed.csv').set_index(keys).sort_index()
    assert len(per_seed)==320
    np.testing.assert_allclose(actual,per_seed[list(METRICS)],rtol=0,atol=1e-12,equal_nan=True)
    summary=pd.read_csv(folder/'summary.csv')
    assert len(summary)==280 and not summary.duplicated(['method','stratum_name','metric']).any()
    original=pd.read_csv(ROOT/'results/stratification/summary.csv')
    metric_map={'overall':'overall','stratum':'cpa','matched':'matched','stratum_minus_overall':'cpa_minus_overall','stratum_minus_matched':'cpa_minus_matched'}
    valid_summaries=0
    for mi,method in enumerate(METHODS):
        for group in STRATA:
            part=per_seed.xs((method,group),level=('method','stratum_name')).sort_index()
            for metric in METRICS:
                ref=summary[(summary.method==method)&(summary.stratum_name==group)&(summary.metric==metric)].iloc[0]
                values=part[metric].to_numpy()
                if np.isnan(values).all():
                    assert ref.n_seeds==0 and ref[['mean','ci_low','ci_high']].isna().all()
                    continue
                assert np.isfinite(values).all() and ref.n_seeds==8
                result=bootstrap(values,np.random.default_rng(613+mi));valid_summaries+=1
                np.testing.assert_allclose(result,[ref['mean'],ref.ci_low,ref.ci_high],rtol=0,atol=1e-12)
                if group=='within_10s' and metric in metric_map:
                    ref=original[(original.method==method)&(original.metric==metric_map[metric])].iloc[0]
                    np.testing.assert_allclose(result,[ref['mean'],ref.ci_low,ref.ci_high],rtol=0,atol=1e-12)
    support=pd.read_csv(folder/'support.csv')
    assert len(support)==40 and not support.duplicated(['task','stratum']).any()
    np.testing.assert_allclose(support.fraction,support.stratum_n/support.n,rtol=0,atol=1e-12)
    for task,part in support.groupby('task'):
        s=part.set_index('stratum');assert (s.n==2000).all()
        assert s.loc[list(BANDS)].stratum_n.sum()==2000
        assert s.loc[list(INSIDE_BANDS)].stratum_n.sum()==s.loc['within_10s','stratum_n']
        assert s.loc['ongoing','stratum_n']+s.loc['new_0_3s','stratum_n']==s.loc['within_3s','stratum_n']
        assert s.loc['within_3s','stratum_n']+s.loc['new_3_5s','stratum_n']==s.loc['within_5s','stratum_n']
        assert s.loc['within_5s','stratum_n']+s.loc['new_5_10s','stratum_n']==s.loc['within_10s','stratum_n']
        assert s.loc['no_entry_10s','overlap_cpa_n']==0
    for (_,_,task),part in frame.groupby(['method','seed','task']):
        s=part.set_index('stratum_name')
        ref=support[support.task==task].set_index('stratum')
        np.testing.assert_array_equal(s.stratum_n.sort_index(),ref.stratum_n.sort_index())
        for band_names in [BANDS,INSIDE_BANDS]:
            band=s.loc[list(band_names)];weights=band.stratum_n.to_numpy();weights=weights/weights.sum()
            expected=s.loc['within_10s','stratum'] if band_names==INSIDE_BANDS else s.overall.iloc[0]
            np.testing.assert_allclose(weights@band.stratum.to_numpy(),expected,rtol=0,atol=1e-12)
            matched=s.loc['within_10s','matched'] if band_names==INSIDE_BANDS else expected
            np.testing.assert_allclose(weights@band.matched.to_numpy(),matched,rtol=0,atol=1e-12)
            if band_names==INSIDE_BANDS:
                np.testing.assert_allclose(weights@band.cpa_matched.to_numpy(),expected,rtol=0,atol=1e-12)
    assert valid_summaries==240
    return {'ttc_task_rows':1600,'ttc_seed_rows':320,'finite_summary_interval_pairs':valid_summaries,
            'structural_undefined_summary_rows':40,'tte_10s_matches_primary':'PASS',
            'weighted_band_reconstruction':'PASS','geometry':check_geometry()}


if __name__=='__main__':
    print(json.dumps(check_ttc_metrics(),indent=2))

import json, sys, statistics as st
rows = []
for p in sys.argv[1:]:
    with open(p) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append(d)
ids = [r['graph_id'] for r in rows]
print('records', len(rows), 'unique graph_ids', len(set(ids)))
print('schemas', set(r['schema'] for r in rows))
print('eager_full_surface_compilation', set(str(r.get('eager_full_surface_compilation')) for r in rows))
def g(r, phase, k):
    return r[phase]['stats'][k]
cols = {}
for r in rows:
    init_c = g(r,'cache_after_initialization','kernel_compile_seconds') + g(r,'cache_after_initialization','evaluate_seconds')
    onl_c = g(r,'cache_after_online_learning','kernel_compile_seconds') + g(r,'cache_after_online_learning','evaluate_seconds') - init_c
    aud_c = g(r,'cache_after_independent_audit','kernel_compile_seconds') + g(r,'cache_after_independent_audit','evaluate_seconds') - init_c - onl_c
    onl_s = g(r,'cache_after_online_learning','step_sample_seconds') - g(r,'cache_after_initialization','step_sample_seconds')
    aud_s = g(r,'cache_after_independent_audit','step_sample_seconds') - g(r,'cache_after_online_learning','step_sample_seconds')
    vals = dict(
        elapsed=r['elapsed_seconds'],
        baseline=r['baseline_calibration_seconds'],
        online=r['concurrent_online_learning_seconds'],
        audit=r['post_learning_audit_seconds'],
        compile_init=init_c, compile_online=onl_c, compile_audit=aud_c,
        sample_online=onl_s, sample_audit=aud_s,
    )
    vals['other'] = vals['elapsed'] - vals['baseline'] - vals['online'] - vals['audit']
    for k, v in vals.items():
        cols.setdefault(k, []).append(v)
print(f"{'phase':16s} {'total_h':>9s} {'median_s':>10s} {'min_s':>9s} {'max_s':>10s} {'share':>6s}")
tot = sum(cols['elapsed'])
for k, v in cols.items():
    print(f"{k:16s} {sum(v)/3600:9.2f} {st.median(v):10.1f} {min(v):9.1f} {max(v):10.1f} {sum(v)/tot:6.1%}")

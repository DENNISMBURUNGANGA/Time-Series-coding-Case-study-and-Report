import numpy as np, pandas as pd

HORIZON = 24

def build_lags(s, lags, horizon=HORIZON):
    out = pd.DataFrame(index=s.index)
    for L in lags:
        assert L >= horizon, 'lag %d < horizon %d leaks' % (L, horizon)
        out['lag_%d' % L] = s.shift(L)
    out['roll_mean_24'] = s.shift(horizon).rolling(24).mean()
    return out

def test_lags_never_shorter_than_horizon():
    idx = pd.date_range('2020-01-01', periods=500, freq='h')
    build_lags(pd.Series(np.arange(500.0), index=idx), [24, 48, 168])

def test_future_does_not_affect_past():
    idx = pd.date_range('2020-01-01', periods=500, freq='h')
    s = pd.Series(np.arange(500.0), index=idx)
    cut = idx[400]
    s2 = s.copy(); s2.loc[cut:] = s2.loc[cut:] * 10 + 5000
    a = build_lags(s,  [24, 48, 168]).loc[:cut - pd.Timedelta(hours=1)]
    b = build_lags(s2, [24, 48, 168]).loc[:cut - pd.Timedelta(hours=1)]
    assert (a.fillna(-1) == b.fillna(-1)).all().all()

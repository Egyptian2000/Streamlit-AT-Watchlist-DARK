
import pandas as pd
import numpy as np

# Column sets
COMPONENT_COLS = ["AC_ShortFlags","AD_Management","AE_Governance","AF_BizModel","AG_TopDown",
                  "AH_Competition","AI_Industry","AJ_Optionality","AK_Custom1","AL_Custom2","AM_Custom3"]

def normalize_weights(w):
    s = sum(w.values())
    if s == 0:
        k = 1.0/len(w)
        return {k_: k for k_ in w}
    return {k_: (v/s) for k_, v in w.items()}

def compute_pw_irr(cases_df):
    # cases_df: asset_id, case_label, prob, irr
    tmp = cases_df.copy()
    tmp["prod"] = tmp["prob"].astype(float) * tmp["irr"].astype(float)
    pw = tmp.groupby("asset_id")["prod"].sum()
    return pw

def compute_confidence(scores_df, comp_weights, k_offset=-40.0):
    # scores_df: index=asset_id, columns COMPONENT_COLS
    # comp_weights: dict of COMPONENT_COLS -> weight (should sum to 1)
    # K = MAX(1, SUMPRODUCT(scores, weights)*20 + k_offset) / (100 + k_offset) * 100
    w = pd.Series(comp_weights)
    # Align missing cols to 0
    for c in w.index:
        if c not in scores_df.columns:
            scores_df[c] = 0.0
    raw = (scores_df[w.index] * w).sum(axis=1)
    scaled = np.maximum(1.0, raw*20.0 + k_offset)
    K = scaled / (100.0 + k_offset) * 100.0
    return K

def ranks_and_blend(J, K):
    # M = rank_avg(J desc), N = rank_avg(K desc)
    M = J.rank(method="average", ascending=False)
    N = K.rank(method="average", ascending=False)
    O = M.max() - M + 1
    P = N.max() - N + 1
    Q = (O + P) / 2.0
    # R = rank_eq(Q desc) -> use method="min" to emulate RANK.EQ ties
    R = Q.rank(method="min", ascending=False)
    return M, N, O, P, Q, R

def compute_weights(Q, R, top_n=12, min_pos=0.03, max_pos=0.10):
    Q = Q.copy()
    R = R.copy()
    is_top = R <= int(top_n)
    q_top = Q[is_top]
    out = pd.Series(0.0, index=Q.index, dtype=float)
    if q_top.empty:
        return out
    minQ, maxQ = q_top.min(), q_top.max()
    span = float(maxQ - minQ)
    delta = float(max_pos - min_pos)
    if span == 0.0:
        out[is_top] = float(max_pos)
        return out
    out[is_top] = min_pos + (Q[is_top] - minQ) * delta / span
    return out

def normalize_portfolio(weights):
    total = weights.sum()
    if total > 0:
        return weights / total
    return weights

def run_model(assets_df, cases_df, comp_weights, top_n, min_pos, max_pos, k_offset, normalize=False):
    # assets_df columns: asset_id, asset_name, COMPONENT_COLS (scores)
    # cases_df columns: asset_id, case_label, prob, irr
    comp_weights = normalize_weights(comp_weights)

    # J
    J = compute_pw_irr(cases_df).reindex(assets_df["asset_id"]).fillna(0.0)
    # K
    score_cols = [c for c in COMPONENT_COLS if c in assets_df.columns]
    scores = assets_df.set_index("asset_id")[score_cols].fillna(0.0)
    K = compute_confidence(scores, comp_weights, k_offset=k_offset)

    # Ranks & blends
    M, N, O, P, Q, R = ranks_and_blend(J, K)

    # Weights
    W = compute_weights(Q, R, top_n=top_n, min_pos=min_pos, max_pos=max_pos)
    W_norm = normalize_portfolio(W) if normalize else W

    # Assemble outputs
    result = assets_df[["asset_id","asset_name"]].copy().set_index("asset_id")
    result["J_PW_IRR"] = J
    result["K_Confidence"] = K
    result["M_RankJ"] = M
    result["N_RankK"] = N
    result["O_InvRankJ"] = O
    result["P_InvRankK"] = P
    result["Q_Blend"] = Q
    result["R_RankBlend"] = R
    result["Weight"] = W
    result["Weight_Norm"] = W_norm
    result = result.reset_index()
    return result

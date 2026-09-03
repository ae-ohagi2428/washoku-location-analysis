"""
05. 重回帰で分析する
  価格帯別（安価／中間／高級）に、同じ説明変数で3本の回帰を行う。
  価格帯別の3本の式において同じ説明変数を使用するのは、価格帯で効く要因がどう変わるかを比べるため。

出すもの:
  (1) 分布の確認        歪度を見て対数変換の妥当性を確かめる
  (2) 相関とVIF         説明変数どうしの重複を確認する
  (3) 段階的な重回帰    変数群を足すごとに R² がどれだけ改善するか
  (4) 係数              偏回帰係数・p値（他の変数を一定にしたときの効果）
  (5) 効果量            β・ΔR²・f²（どの変数がどれだけ効いているか）
  (6) 残差の大きさ      実際の店舗数で何店ずれるか
  (7) 予測値を下回る駅  予測より少ない駅＝出店余地の候補

入力:  station_dataset.csv
出力:  regression_result.csv（駅ごとの予測値と残差）

実行:
  pip install pandas numpy statsmodels
  python 05_analyze.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

CSV = "data/station_dataset.csv"
OUT = "output/regression_result.csv"

TARGETS = {"casual": "安価帯（〜5000円）",
           "middle": "中間帯（5001〜10000円）",
           "luxury": "高級帯（10001円〜）"}

# 説明変数6つ。多重共線性の程度は (2) で相関とVIFを出して毎回確認する。
VARS = ["passengers", "n_stations_1500m", "estab_food_1500m",
        "estab_hotel_1500m", "land_price_commercial", "pop_1500m"]

# 分布が右に大きく歪む変数・桁数が多い変数は対数変換する（0があるので +1）。
# 駅の密集度のみ0〜16程度、歪みも少ないため対数変換しない。
LOG_VARS = ["passengers", "estab_food_1500m", "estab_hotel_1500m",
            "land_price_commercial", "pop_1500m"]

# 変数群を足していく順番
STEPS = [("1 乗降客数のみ", ["passengers"]),
         ("2 + 駅の密集度", ["n_stations_1500m"]),
         ("3 + 商業集積",   ["estab_food_1500m", "estab_hotel_1500m"]),
         ("4 + 地価",       ["land_price_commercial"]),
         ("5 + 夜間人口",   ["pop_1500m"])]


def prepare(df):
    X = pd.DataFrame(index=df.index)
    for c in VARS:
        if c not in df.columns:
            print(f"[!] 列がありません: {c}"); continue
        v = df[c].astype(float)
        X[c] = np.log10(v.clip(lower=0) + 1) if c in LOG_VARS else v
    return X


def section(title):
    print(f"\n{'='*64}\n{title}\n{'='*64}")


def check_data(df, X):
    # ---- (1) 分布の確認 ----
    section("分布の確認")
    cols = [c for c in VARS if c in df.columns] + list(TARGETS)
    print(df[cols].describe().round(1).to_string())
    print("\n  歪度（0に近いほど左右対称。2を超えると強く右に歪む）")
    for c in cols:
        print(f"    {c:24}{df[c].skew():7.2f}")

    # ---- (2) 相関とVIF ----
    section("相関とVIF")
    print("  目的変数との相関")
    print("  ※ (log) 印は log10(x+1) 変換済み。目的変数も log10(x+1)")
    print("  変数" + " " * 26 + f"{'casual':>8}{'middle':>8}{'luxury':>8}")
    for c in X.columns:
        name = f"{c} (log)" if c in LOG_VARS else c
        row = f"    {name:28}"
        for t in TARGETS:
            row += f"{X[c].corr(np.log10(df[t] + 1)):8.3f}"
        print(row)

    corr = X.corr()
    pairs = [(abs(corr.loc[a, b]), a, b, corr.loc[a, b])
             for i, a in enumerate(X.columns) for b in X.columns[i+1:]
             if abs(corr.loc[a, b]) >= 0.7]
    print("\n  説明変数どうしで相関が高い組（|r| >= 0.7）")
    if pairs:
        for _, a, b, r in sorted(pairs, reverse=True):
            print(f"    {r:6.3f}  {a} × {b}")
    else:
        print("    なし")

    Xc = X.dropna().assign(_c=1.0)
    rows = [(c, vif(Xc.to_numpy(), i)) for i, c in enumerate(Xc.columns) if c != "_c"]
    print("\n  VIF（10以上は他の変数と強く重複、5以上は要注意）")
    for c, v in sorted(rows, key=lambda r: -r[1]):
        name = f"{c} (log)" if c in LOG_VARS else c
        print(f"    {name:28}{v:7.2f}")


def analyze(df, X, tgt, label, out):
    y = np.log10(df[tgt].astype(float) + 1)
    zero = (df[tgt] == 0).sum()
    section(f"■ {label}  合計 {int(df[tgt].sum())} 店 / "
            f"ゼロ {zero} 駅（{zero/len(df)*100:.1f}%）")

    # ---- (3) 段階的な重回帰 ----
    print("-- モデルの積み上げ --")
    cols, prev = [], None
    for name, add in STEPS:
        cols += [c for c in add if c in X.columns]
        m = sm.OLS(y, sm.add_constant(X[cols])).fit()
        d = "" if prev is None else f"  (調整済み {m.rsquared_adj-prev:+.3f})"
        print(f"  {name:16} 変数{len(cols):>2}個  R²={m.rsquared:.3f}  "
              f"調整済み={m.rsquared_adj:.3f}{d}")
        prev = m.rsquared_adj
    full = m

    # ---- (4) 係数 ----
    print(f"\n-- フルモデルの係数（n={int(full.nobs)}）--")
    print("  変数                     係数     標準誤差   p値      判定")
    for c in cols:
        p = full.pvalues[c]
        sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        print(f"  {c:22}{full.params[c]:9.3f}{full.bse[c]:10.3f}{p:10.3g}   {sig}")
    print(f"  {'(定数)':22}{full.params['const']:9.3f}{full.bse['const']:10.3f}")
    print("  *** p<0.001  ** p<0.01  * p<0.05  n.s. 有意でない")

    # ---- (5) 効果量 ----
    Xz = (X[cols] - X[cols].mean()) / X[cols].std(ddof=1)
    std = sm.OLS((y - y.mean()) / y.std(ddof=1), sm.add_constant(Xz)).fit()
    print("\n-- 効果量 --")
    print("  変数                       β      ΔR²      f²    判定")
    res = []
    for c in cols:
        sub = sm.OLS(y, sm.add_constant(X[cols].drop(columns=[c]))).fit()
        d_r2 = full.rsquared - sub.rsquared
        res.append((c, std.params[c], d_r2, d_r2 / (1 - full.rsquared)))
    for c, b, d, f2 in sorted(res, key=lambda r: -abs(r[1])):
        size = "大" if f2 >= .35 else "中" if f2 >= .15 else "小" if f2 >= .02 else "－"
        print(f"  {c:22}{b:7.3f}{d:8.3f}{f2:8.3f}    {size}")
    print("  β: 1標準偏差あたりの効果（変数どうしを比べられる）")
    print("  ΔR²: その変数を抜いたときのR²の落ち込み  f²: 0.02小/0.15中/0.35大")

    print("\n" + "-" * 64)
    print("  ここまででモデルが完成。ここからは各駅の実測値と見比べる。")
    print("  実測値 − 予測値 ＝ 残差。この負の大きい駅が出店余地の候補になる。")

    # ---- (6) 残差の大きさ ----
    pred = (10 ** full.fittedvalues - 1).clip(lower=0)
    err = df[tgt].astype(float) - pred
    sigma = full.resid.std(ddof=len(full.params))
    print(f"\n-- 残差の大きさ --")
    print(f"  [実数] 残差の中央値 {err.abs().median():.1f} 店 / 平均 {err.abs().mean():.1f} 店")
    print(f"  [実数] ±1店以内 {(err.abs()<=1).mean()*100:.1f}%  "
          f"±2店以内 {(err.abs()<=2).mean()*100:.1f}%")
    print(f"  [対数] 残差の標準偏差 {sigma:.3f} "
          f"→ 実測は予測値の {1/10**sigma:.2f}〜{10**sigma:.2f}倍 に約7割が収まる")

    # ---- (7) 予測値を下回る駅 ----
    out[f"pred_{tgt}"] = pred.round(1)
    out[f"resid_{tgt}"] = full.resid.round(3)
    print(f"\n-- 予測値を下回る駅 上位10 --")
    tmp = out.assign(actual=df[tgt], food=df["estab_food_1500m"])
    for i, (_, r) in enumerate(tmp.nsmallest(10, f"resid_{tgt}").iterrows(), 1):
        print(f"  {i:>2}. {r['station_name']:　<11} 実際{int(r['actual']):>3}店 "
              f"予測値{r[f'pred_{tgt}']:>5.1f}店  残差(対数){r[f'resid_{tgt}']:+.3f}  "
              f"乗降{int(r['passengers']):>8,} 飲食店{int(r['food']):>4}")


def main():
    df = pd.read_csv(CSV)
    X = prepare(df)
    print(f"■ 駅: {len(df)} 駅 / 説明変数: {len(X.columns)} 個")
    print("  ※ 目的変数と、歪んだ説明変数は log10(x+1) で変換")

    check_data(df, X)
    out = df[["station_name", "passengers", "shop_count"]].copy()
    for tgt, label in TARGETS.items():
        analyze(df, X, tgt, label, out)

    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT} に予測値と残差を出力")


if __name__ == "__main__":
    main()

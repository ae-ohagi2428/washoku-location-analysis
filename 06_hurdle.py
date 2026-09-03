"""
06. 二段階モデル（ハードルモデル）で出店余地を探す
  線形重回帰モデルでは、残差の下位が「実際0店」の駅で埋まってしまう結果になった。
  予測値3店に対して実測0という駅が、対数スケールで最も大きな負になるためで、
  「ゼロ」と「少ない」を1つの尺度で測っていることの帰結にあたる。

  そこで過程を2つに分ける。

  第1段階  全駅を対象に「1店以上あるか」をロジスティック回帰で予測する。
           予測確率が高いのに実際は0店である駅を算出する。

  第2段階  1店以上ある駅だけで店舗数を回帰する。

入力:  station_dataset.csv
出力:  hurdle_result.csv（第1段階の確率、第2段階の残差、候補フラグ）

実行:
  pip install pandas numpy statsmodels
  python 06_hurdle.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

CSV = "data/station_dataset.csv"
OUT = "output/hurdle_result.csv"

TARGETS = {"casual": "安価帯（〜5000円）",
           "middle": "中間帯（5001〜10000円）",
           "luxury": "高級帯（10001円〜）"}

VARS = ["passengers", "n_stations_1500m", "estab_food_1500m",
        "estab_hotel_1500m", "land_price_commercial", "pop_1500m"]
LOG_VARS = ["passengers", "estab_food_1500m", "estab_hotel_1500m",
            "land_price_commercial", "pop_1500m"]

PROB_HIGH = 0.5     # この確率を超えるのに0店なら「空白」の候補とみなす


def prepare(df):
    X = pd.DataFrame(index=df.index)
    for c in VARS:
        v = df[c].astype(float)
        X[c] = np.log10(v.clip(lower=0) + 1) if c in LOG_VARS else v
    return X


def stage1(df, X, tgt, out):
    """第1段階：1店以上あるかをロジスティック回帰で予測する。"""
    y = (df[tgt] > 0).astype(int)
    Xc = sm.add_constant(X)
    m = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
    if not m.mle_retvals["converged"]:
        print("  [!] 収束していません。以下の結果は信用できません")

    print(f"\n-- 第1段階：1店以上あるか（n={len(y)}, 1店以上={int(y.sum())}駅）--")
    # 疑似決定係数。通常のR²と違い、0.2〜0.4でも当てはまりは良いとされる
    print(f"  McFadden 疑似R² = {m.prsquared:.3f}")
    print("  変数                     係数    オッズ比    p値     判定")
    for c in X.columns:
        p = m.pvalues[c]
        sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        # オッズ比 = その変数が1単位増えたとき、
        #            「1店以上ある見込み／ない見込み」の比が何倍になるか
        print(f"  {c:22}{m.params[c]:9.3f}{np.exp(m.params[c]):10.2f}"
              f"{p:10.3g}   {sig}")

    print("\n  -- 各駅の実測データに当てはめ、予測の正誤を確認 --")
    prob = m.predict(Xc)
    pred = (prob >= PROB_HIGH).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    print(f"\n  的中率 {acc*100:.1f}%"
          f"   （1店以上を当てた {tp} / 予測0→実際1以上 {fn}"
          f" / ゼロを当てた {tn} / 予測1以上→実際0 {fp}）")

    # ベースライン＝全駅を多数派と予測した場合の的中率。
    # 多数派の割合が高いほどベースラインは高く出るため、
    # 的中率とベースラインとの差を比べる必要がある。
    base = max(y.mean(), 1 - y.mean())
    maj = "1店以上" if y.mean() >= 0.5 else "0店"
    print(f"  ベースライン（全駅を「{maj}」と予測した場合） {base*100:.1f}%"
          f"   差 {(acc - base)*100:+.1f}ポイント")

    out[f"prob_{tgt}"] = prob.round(3)
    return prob


def stage2(df, X, tgt, out):
    """第2段階：1店以上ある駅だけで店舗数を回帰する。"""
    sub = df[tgt] > 0
    y = np.log10(df.loc[sub, tgt].astype(float))
    m = sm.OLS(y, sm.add_constant(X[sub])).fit()

    # 標準化偏回帰係数（β）。1店以上の駅の中で標準化する。
    Xz = (X[sub] - X[sub].mean()) / X[sub].std(ddof=1)
    std = sm.OLS((y - y.mean()) / y.std(ddof=1), sm.add_constant(Xz)).fit()

    print(f"\n-- 第2段階：店舗数（1店以上の {int(sub.sum())} 駅のみ）--")
    print(f"  R² = {m.rsquared:.3f}  調整済み = {m.rsquared_adj:.3f}")
    print("  変数                     係数    標準誤差       β    p値     判定")
    for c in X.columns:
        p = m.pvalues[c]
        sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        print(f"  {c:22}{m.params[c]:9.3f}{m.bse[c]:10.3f}{std.params[c]:9.3f}"
              f"{p:10.3g}   {sig}")

    out.loc[sub, f"pred2_{tgt}"] = (10 ** m.fittedvalues).round(1)
    out.loc[sub, f"resid2_{tgt}"] = m.resid.round(3)
    return m


def main():
    df = pd.read_csv(CSV)
    X = prepare(df)
    out = df[["station_name", "passengers", "estab_food_1500m"]].copy()
    print(f"■ 駅: {len(df)} 駅 / 説明変数: {len(X.columns)} 個")

    for tgt, label in TARGETS.items():
        zero = int((df[tgt] == 0).sum())
        print(f"\n{'='*64}\n■ {label}  "
              f"合計 {int(df[tgt].sum())} 店 / ゼロ {zero} 駅"
              f"（{zero/len(df)*100:.1f}%）\n{'='*64}")

        print(f"\n① モデル推定 {'━'*50}")
        prob = stage1(df, X, tgt, out)
        stage2(df, X, tgt, out)

        print(f"\n② 出店候補駅 {'━'*50}")
        # ---- 候補1：条件は揃っているのに空白 ----
        blank = df[(df[tgt] == 0) & (prob >= PROB_HIGH)].index
        out[f"blank_{tgt}"] = 0
        out.loc[blank, f"blank_{tgt}"] = 1
        print(f"\n-- 候補A：実際は0店だが、1店以上あると予測された駅（見込み(確率)10位まで）--")
        if len(blank):
            t = out.loc[blank].assign(p=prob[blank]).nlargest(10, "p")
            for i, (_, r) in enumerate(t.iterrows(), 1):
                print(f"  {i:>2}. {r['station_name']:　<11} 見込み{r['p']*100:>5.1f}%"
                      f"  乗降{int(r['passengers']):>8,}"
                      f"  飲食店{int(r['estab_food_1500m']):>4}")
        else:
            print("  該当なし")

        # ---- 候補2：出店済みだが予測値より少ない ----
        col = f"resid2_{tgt}"
        if col in out.columns and out[col].notna().any():
            print(f"\n-- 候補B：実際に1店以上あるが、予測値より少ない駅（残差の小さい順10駅）--")
            t = out[out[col].notna()].nsmallest(10, col).assign(actual=df[tgt])
            for i, (_, r) in enumerate(t.iterrows(), 1):
                print(f"  {i:>2}. {r['station_name']:　<11} 実際{int(r['actual']):>3}店"
                      f" 予測値{r[f'pred2_{tgt}']:>5.1f}店  残差(対数){r[col]:+.3f}"
                      f"  飲食店{int(r['estab_food_1500m']):>4}")

    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT} に出力")
    print("  prob_*   1店以上ある見込み（第1段階）")
    print("  blank_*  見込みが高いのに0店＝候補A")
    print("  pred2_*  出店済みの駅での予測値（第2段階）")
    print("  resid2_* 出店済みの駅での予測値とのずれ（第2段階）＝候補B")

if __name__ == "__main__":
    main()

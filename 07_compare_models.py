"""
07. 負の二項回帰と比べる（頑健性の確認）
  和食店数は「0, 1, 2, 3…」というカウントデータなので、
  本来はカウント向けのモデルを使う考え方もある。
  現行の線形重回帰モデルと結果がどれだけ違うかを確かめる。

  ポアソン回帰ではなく負の二項回帰を使う理由:
    ポアソン回帰は「分散＝平均」を仮定するが、今回のデータは分散が平均の
    8〜11倍あり（過分散）、この仮定が成り立たない。
    負の二項回帰はばらつきの大きさを表すパラメータ α を別に持つので使用可能。

  見るのは係数そのものより、算出された候補駅が異なるかどうか。
  同じであれば「手法を変えても結論は同じ」と言える。

入力:  station_dataset.csv
出力:  model_comparison.csv（両モデルの予測値と残差、順位）

実行:
  pip install pandas numpy statsmodels scipy
  python 07_compare_models.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

CSV = "data/station_dataset.csv"
OUT = "output/model_comparison.csv"
TOP_N = 15          # 比較したい候補駅の駅数

TARGETS = {"casual": "安価帯（〜5000円）",
           "middle": "中間帯（5001〜10000円）",
           "luxury": "高級帯（10001円〜）"}
VARS = ["passengers", "n_stations_1500m", "estab_food_1500m",
        "estab_hotel_1500m", "land_price_commercial", "pop_1500m"]
LOG_VARS = ["passengers", "estab_food_1500m", "estab_hotel_1500m",
            "land_price_commercial", "pop_1500m"]


def prepare(df):
    X = pd.DataFrame(index=df.index)
    for c in VARS:
        v = df[c].astype(float)
        X[c] = np.log10(v.clip(lower=0) + 1) if c in LOG_VARS else v
    return X


def main():
    df = pd.read_csv(CSV)
    X = prepare(df)
    Xc = sm.add_constant(X)
    out = df[["station_name", "passengers", "estab_food_1500m"]].copy()
    print(f"■ 駅: {len(df)} 駅 / 説明変数: {len(X.columns)} 個\n")

    for tgt, label in TARGETS.items():
        y = df[tgt].astype(float)
        print("=" * 66)
        print(f"■ {label}  合計 {int(y.sum())} 店 / "
              f"ゼロ {int((y==0).sum())} 駅")
        print("=" * 66)

        # ---- 過分散の確認 ----
        print(f"-- 過分散の確認 --")
        print(f"  平均 {y.mean():.2f} / 分散 {y.var():.2f} / "
              f"分散÷平均 = {y.var()/y.mean():.1f}")
        print("  ※ 1に近ければポアソン回帰でよい。大きいほど負の二項が必要")

        # ---- 現行：対数変換した線形回帰 ----
        ols = sm.OLS(np.log10(y + 1), Xc).fit()
        pred_ols = (10 ** ols.fittedvalues - 1).clip(lower=0)
        resid_ols = ols.resid

        # ---- 負の二項回帰 ----
        try:
            nb = sm.NegativeBinomial(y, Xc).fit(disp=0, maxiter=500)
            ok = True
        except Exception as e:
            print(f"\n  [!] 負の二項回帰が収束しませんでした: {e}")
            ok = False
        if not ok:
            print()
            continue
        if not nb.mle_retvals.get("converged", False):
            print("\n  [!] 収束していません。結果は参考値として扱ってください")

        alpha = float(nb.params.iloc[-1])
        mu = nb.predict(Xc)
        # ピアソン残差。カウントモデルでは分散が平均に応じて変わるので、
        # 単純な差ではなく、ばらつきの大きさで割って比べる。
        resid_nb = (y - mu) / np.sqrt(mu + alpha * mu ** 2)

        print(f"\n-- 係数の比較 --")
        print("  変数                   線形重回帰  判定    負の二項  判定    符号")
        def stars(p):
            """p値から判定記号を返す。NaN（算出不可）は n.s. と区別する。"""
            if pd.isna(p):
                return "--"
            return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

        for c in X.columns:
            a, b = ols.params[c], nb.params[c]
            sa, sb = stars(ols.pvalues[c]), stars(nb.pvalues[c])
            same = "一致" if np.sign(a) == np.sign(b) else "★不一致"
            print(f"  {c:22}{a:9.3f} {sa:6}{b:9.3f} {sb:6}{same}")
        print("  ※ -- は標準誤差が算出できず、p値が出せなかったもの")

        # ---- 算出された候補駅を比較する ----
        r_ols = pd.Series(resid_ols).rank()
        r_nb = pd.Series(resid_nb).rank()
        rho = spearmanr(r_ols, r_nb).statistic
        top_ols = set(pd.Series(resid_ols).nsmallest(TOP_N).index)
        top_nb = set(pd.Series(resid_nb).nsmallest(TOP_N).index)
        share = len(top_ols & top_nb)

        print(f"\n-- 候補駅（予測より少ない駅）の比較 --")
        print(f"  順位の相関（スピアマン）: {rho:.3f}")
        print(f"  上位{TOP_N}駅の重なり: {share} / {TOP_N} 駅"
              f"（{share/TOP_N*100:.0f}%）")

        print(f"\n  順位  {'線形重回帰':　<12}{'負の二項'}")
        o = pd.Series(resid_ols).nsmallest(TOP_N).index
        n = pd.Series(resid_nb).nsmallest(TOP_N).index
        for i in range(TOP_N):
            a = df["station_name"].iloc[o[i]]
            b = df["station_name"].iloc[n[i]]
            mark = "" if a == b else ("  ←両方に登場" if o[i] in top_nb else "")
            print(f"  {i+1:>3}.  {a:　<12}{b:　<12}{mark}")

        out[f"pred_ols_{tgt}"] = pred_ols.round(1)
        out[f"resid_ols_{tgt}"] = resid_ols.round(3)
        out[f"pred_nb_{tgt}"] = mu.round(1)
        out[f"resid_nb_{tgt}"] = resid_nb.round(3)
        print()

    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"→ {OUT} に出力")
    print("  resid_ols_* 線形重回帰の残差 / resid_nb_* 負の二項のピアソン残差")


if __name__ == "__main__":
    main()

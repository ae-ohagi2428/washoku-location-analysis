"""
08. 交差検証（汎化性能の確認）
  05・06 で算出した R² や的中率は、
  係数を決めるために使用したデータと同じ駅のデータで測定しているため、
  当てはまりが良いというのは当然の結果である。

  そこで駅データを5個に分け、そのうち4個を使って係数を決めて
  残り1個のデータで予測値を測定する、という手順を
  分割方法を変えながら繰り返す。
  学習に使っていない駅で検証を行い、検証と学習の差を確認することで、
  「他の駅にも通用するモデルか」が分かる。

  学習と検証の差が小さければ安定し、大きければ過学習の可能性がある。
  その場合、算出された候補駅の差についても、データの偶然に左右されている可能性がある。

  ※ 5分割を10回繰り返して平均することで、5分割時の偏りをならす。
     第1段階は0店/1店以上の比率を分割ごとに保つ（層化）。

入力:  station_dataset.csv
出力:  cross_validation.csv（学習・検証のスコアと標準偏差、ベースライン）

実行:
  pip install pandas numpy statsmodels scikit-learn
  python 08_cross_validate.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

CSV = "data/station_dataset.csv"
OUT = "output/cross_validation.csv"

TARGETS = {"casual": "安価帯（〜5000円）",
           "middle": "中間帯（5001〜10000円）",
           "luxury": "高級帯（10001円〜）"}

VARS = ["passengers", "n_stations_1500m", "estab_food_1500m",
        "estab_hotel_1500m", "land_price_commercial", "pop_1500m"]
LOG_VARS = ["passengers", "estab_food_1500m", "estab_hotel_1500m",
            "land_price_commercial", "pop_1500m"]

N_SPLITS  = 5
N_REPEATS = 10
SEED      = 0
MIN_N     = 50      # 各価格帯におけるデータ数がこれを下回る場合は分割数を減らす


def prepare(df):
    X = pd.DataFrame(index=df.index)
    for c in VARS:
        v = df[c].astype(float)
        X[c] = np.log10(v.clip(lower=0) + 1) if c in LOG_VARS else v
    return X


def r2(y, pred):
    """決定係数。検証側では学習側の平均は使用せず、別途計算する。"""
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def n_splits_for(n):
    """駅数が少ないときは分割数を減らす（検証側が小さくなりすぎないように）。"""
    if n >= MIN_N:
        return N_SPLITS
    return max(2, min(N_SPLITS, n // 10))


def cv_ols(X, y):
    """線形重回帰モデル を交差検証する。戻り値は学習・検証それぞれの R²。"""
    Xc = sm.add_constant(X, has_constant="add").to_numpy(float)
    yv = y.to_numpy(float)
    k = n_splits_for(len(yv))
    kf = RepeatedKFold(n_splits=k, n_repeats=N_REPEATS, random_state=SEED)

    tr_s, te_s = [], []
    for tr, te in kf.split(Xc):
        m = sm.OLS(yv[tr], Xc[tr]).fit()
        tr_s.append(r2(pd.Series(yv[tr]), m.predict(Xc[tr])))
        te_s.append(r2(pd.Series(yv[te]), m.predict(Xc[te])))
    return np.array(tr_s), np.array(te_s), k


def cv_logit(X, y):
    """ロジスティック回帰を交差検証する。戻り値は学習・検証それぞれの的中率。
    0店/1店以上の比率を分割ごとに保つため層化する。"""
    Xc = sm.add_constant(X, has_constant="add").to_numpy(float)
    yv = y.to_numpy(int)
    # 層化するので、少ないほうのグループが各分割に1件以上入る数までしか割れない
    k = min(n_splits_for(len(yv)), int(min((yv == 0).sum(), (yv == 1).sum())))
    k = max(2, k)
    kf = RepeatedStratifiedKFold(n_splits=k, n_repeats=N_REPEATS,
                                 random_state=SEED)

    tr_s, te_s, n_fail = [], [], 0
    for tr, te in kf.split(Xc, yv):
        try:
            m = sm.Logit(yv[tr], Xc[tr]).fit(disp=0, maxiter=200)
        except Exception:
            n_fail += 1
            continue
        if not m.mle_retvals.get("converged", False):
            n_fail += 1
            continue
        tr_s.append(((m.predict(Xc[tr]) >= .5).astype(int) == yv[tr]).mean())
        te_s.append(((m.predict(Xc[te]) >= .5).astype(int) == yv[te]).mean())
    return np.array(tr_s), np.array(te_s), k, n_fail


def show(name, tr, te, k, extra=""):
    if len(te) == 0:
        print(f"  {name:22} 計算できませんでした")
        return None
    print(f"  {name:22} 学習 {tr.mean():.3f}  検証 {te.mean():.3f}"
          f" (±{te.std():.3f})  差 {tr.mean()-te.mean():+.3f}"
          f"  [{k}分割×{N_REPEATS}回]{extra}")
    return {"train": round(float(tr.mean()), 3),
            "test": round(float(te.mean()), 3),
            "test_sd": round(float(te.std()), 3),
            "gap": round(float(tr.mean() - te.mean()), 3),
            "n_splits": k}


def main():
    df = pd.read_csv(CSV)
    X = prepare(df)
    print(f"■ 駅: {len(df)} 駅 / 説明変数: {len(X.columns)} 個")
    print(f"  {N_SPLITS}分割を{N_REPEATS}回繰り返し、平均で比べる")
    print("  ※ 学習＝係数を決めた駅での値／検証＝使っていない駅での値")
    print("  ※ 差が大きいほど、そのデータに合わせ込みすぎている\n")

    rows = []
    for tgt, label in TARGETS.items():
        n1 = int((df[tgt] > 0).sum())
        print("=" * 70)
        print(f"■ {label}  1店以上 {n1} 駅 / ゼロ {len(df)-n1} 駅")
        print("=" * 70)

        # ---- 05：全駅の線形重回帰 ----
        print("-- 05 全駅の重回帰（対数） R² --")
        y = np.log10(df[tgt].astype(float) + 1)
        tr, te, k = cv_ols(X, y)
        r = show("R²", tr, te, k)
        if r:
            rows.append({"target": tgt, "model": "05_ols_all", **r})

        # ---- 06 第1段階：1店以上あるか ----
        print("\n-- 06 第1段階 ロジスティック回帰 的中率 --")
        yb = (df[tgt] > 0).astype(int)
        # ベースライン＝全駅を多いほうのグループと予測したときの的中率。
        # 的中率はこれを上回って初めて意味を持つので、CSVにも残して図で示す。
        base = max(yb.mean(), 1 - yb.mean())
        maj = "1店以上" if yb.mean() >= 0.5 else "0店"
        tr, te, k, fail = cv_logit(X, yb)
        note = f"  ※収束せず {fail} 回除外" if fail else ""
        r = show("的中率", tr, te, k, note)
        print(f"  （参考）ベースライン（全駅を「{maj}」と予測した場合） {base:.3f}")
        if r:
            print(f"  　　　　検証との差 {r['test'] - base:+.3f}")
            rows.append({"target": tgt, "model": "06_logit", **r,
                         "baseline": round(float(base), 3)})

        # ---- 06 第2段階：1店以上の駅だけ ----
        print("\n-- 06 第2段階 1店以上の駅の重回帰（対数） R² --")
        sub = df[tgt] > 0
        if n1 < 30:
            print(f"  {'R²':22} 駅数が少ないため省略（{n1} 駅）")
        else:
            y2 = np.log10(df.loc[sub, tgt].astype(float))
            tr, te, k = cv_ols(X[sub], y2)
            if n1 < MIN_N:
                print(f"  ※ 1店以上が {n1} 駅と少ないため {k} 分割に落としています")
            r = show("R²", tr, te, k)
            if r:
                rows.append({"target": tgt, "model": "06_ols_positive", **r})
        print()

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"→ {OUT} に出力")
    print("  train 学習に使った駅での値 / test 使っていない駅での値")
    print("  test_sd 検証値のばらつき / gap 学習と検証の差")
    print("  baseline 全駅を多いほうと予測したときの的中率（第1段階のみ）")


if __name__ == "__main__":
    main()

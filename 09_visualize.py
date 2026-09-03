"""
09. 図をつくる
  05〜08 の結果から、記事に貼る図を PNG で書き出す。
  数値はすべて 05〜08 が出したものを読むだけで、ここでは計算し直さない。

つくる図:
  fig1_distribution.png   価格帯別の店舗数の分布  → ゼロが多いことを見せる
  fig2_correlation.png    説明変数どうしの相関    → 変数の重複を見せる
  fig3_residual.png       予測値と残差            → 残差の下位が0店で埋まる
  fig4_validation.png     学習と検証の比較        → 汎化性能

入力:  station_dataset.csv / regression_result.csv / cross_validation.csv
       ※ 05・08 を実行したあとに動かす
出力:  figs/ に PNG 4枚

実行:
  pip install pandas numpy matplotlib japanize-matplotlib
  python 09_visualize.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib      # noqa: F401  読み込むだけで日本語が出る
except ImportError:
    print("[!] japanize-matplotlib が無いので日本語が正しく表示されません")
    print("    pip install japanize-matplotlib")

DATA   = "data/station_dataset.csv"
REG    = "output/regression_result.csv"
CV     = "output/cross_validation.csv"
OUTDIR = "output/figs"

TARGETS = {"casual": "安価帯（〜5000円）",
           "middle": "中間帯（5001〜10000円）",
           "luxury": "高級帯（10001円〜）"}

VARS = ["passengers", "n_stations_1500m", "estab_food_1500m",
        "estab_hotel_1500m", "land_price_commercial", "pop_1500m"]
LOG_VARS = ["passengers", "estab_food_1500m", "estab_hotel_1500m",
            "land_price_commercial", "pop_1500m"]

# 図に出すときの変数名（コード名のままだと読み手に伝わらない）
LABELS = {"passengers": "乗降客数", "n_stations_1500m": "駅の密集度",
          "estab_food_1500m": "飲食店数", "estab_hotel_1500m": "宿泊施設数",
          "land_price_commercial": "商業地地価", "pop_1500m": "夜間人口"}

COLORS = {"casual": "#4C72B0", "middle": "#DD8452", "luxury": "#937860"}


def prepare(df):
    """05・06 と同じ変換をかける（駅の密集度のみ非変換）。"""
    X = pd.DataFrame(index=df.index)
    for c in VARS:
        v = df[c].astype(float)
        X[c] = np.log10(v.clip(lower=0) + 1) if c in LOG_VARS else v
    return X


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------- 図1
def fig_distribution(df):
    """価格帯別に店舗数の分布を描く。ゼロがどれだけ多いかを見せる。"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (tgt, label) in zip(axes, TARGETS.items()):
        v = df[tgt].astype(int)
        zero = int((v == 0).sum())
        top = int(v.quantile(0.99))         # 外れ値で横に伸びすぎないように
        ax.hist(v.clip(upper=top),
            bins=np.arange(-0.5, max(top, 2) + 1.5, 1),
            color=COLORS[tgt], edgecolor="white", linewidth=0.5)
        # 目盛りが多いときは間引く（ラベルが重なるため）
        step = 1 if top <= 12 else 2 if top <= 24 else 5
        ax.set_xticks(range(0, max(top, 2) + 1, step))
        ax.axvline(0.5, color="crimson", linestyle="--", linewidth=1)
        ax.set_title(f"{label}\nゼロ {zero}駅（{zero/len(df)*100:.1f}%）",
                     fontsize=10)
        ax.set_xlabel(f"店舗数（{top}店以上はまとめて表示）", fontsize=9)
        ax.set_ylabel("駅数", fontsize=9)
    fig.suptitle("図1　価格帯別　和食店数の分布", fontsize=12, y=1.04)
    save(fig, "fig1_distribution.png")


# ---------------------------------------------------------------- 図2
def fig_correlation(df):
    """説明変数どうしの相関を色で見せる。多重共線性の確認。"""
    X = prepare(df).rename(columns=LABELS)
    corr = X.corr()
    n = len(corr)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45,
                                                ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(n):
        for j in range(n):
            r = corr.iloc[i, j]
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(r) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="相関係数")
    ax.set_title("図2　説明変数どうしの相関\n（駅の密集度以外は対数変換した後の値）",
                 fontsize=11)
    save(fig, "fig2_correlation.png")


# ---------------------------------------------------------------- 図3
def fig_residual(df, reg):
    """Ⅲ の予測値と残差。0店の駅を色分けして、下側に溜まることを見せる。
    残差が対数なので、横軸も対数の予測値にそろえる。
    予測値の列は実数に戻したあと丸めてあるので、実測値から残差を引いて戻す。"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (tgt, label) in zip(axes, TARGETS.items()):
        resid = reg[f"resid_{tgt}"]
        pred = np.log10(df[tgt].astype(float) + 1) - resid
        zero = df[tgt] == 0
        ax.scatter(pred[~zero], resid[~zero], s=12, alpha=.45,
                   color=COLORS[tgt], label="1店以上", edgecolors="none")
        ax.scatter(pred[zero], resid[zero], s=14, alpha=.7,
                   color="crimson", label="0店", edgecolors="none")
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("予測値  log10(x+1)", fontsize=9)
        ax.set_ylabel("残差（対数）", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("図3　予測値と残差　　赤い点（0店）が下側に溜まっている",
                 fontsize=12, y=1.04)
    save(fig, "fig3_residual.png")


# ---------------------------------------------------------------- 図4
def fig_validation(cv):
    """Ⅵ の学習と検証を並べる。差が小さいほど他の駅にも通用する。
    第一段階の的中率はベースラインを超えて初めて意味を持つので、線で示す。"""
    names = {"05_ols_all": "Ⅲ 全駅の重回帰\n(R²)",
             "06_logit": "Ⅳ 第一段階\n(的中率)",
             "06_ols_positive": "Ⅳ 第二段階\n(R²)"}
    models = [m for m in names if (cv["model"] == m).any()]

    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 3.8),
                             squeeze=False)
    for ax, m in zip(axes[0], models):
        sub = cv[cv["model"] == m].set_index("target").reindex(TARGETS)
        sub = sub[sub["test"].notna()]
        x = np.arange(len(sub))
        ax.bar(x - .2, sub["train"], .38, label="学習", color="#B0B0B0")
        ax.bar(x + .2, sub["test"], .38, yerr=sub["test_sd"], capsize=3,
               label="検証", color="#4C72B0")

        # ベースライン（08 が出した値）。価格帯ごとに高さが違うので線分で引く
        if "baseline" in sub.columns and sub["baseline"].notna().any():
            for xi, b in zip(x, sub["baseline"]):
                if pd.notna(b):
                    ax.hlines(b, xi - .45, xi + .45, color="crimson",
                              linestyle="--", linewidth=1.3, zorder=3)
            ax.plot([], [], color="crimson", linestyle="--", linewidth=1.3,
                    label="ベースライン")

        ax.set_xticks(x)
        ax.set_xticklabels([TARGETS[i].split("（")[0] for i in sub.index],
                           fontsize=9)
        lo = min(0.0, float((sub["test"] - sub["test_sd"]).min()) - .05)
        ax.set_ylim(lo, 1.15)      # 上は凡例のぶんの余白
        if lo < 0:
            ax.axhline(0, color="gray", linewidth=.8)
        ax.set_title(names[m], fontsize=10)
        ax.legend(fontsize=8, loc="upper right", framealpha=.9)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("図4　学習と検証の比較　　差が小さいほど他の駅にも通用する",
                 fontsize=12, y=1.04)
    save(fig, "fig4_validation.png")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(DATA)
    print(f"■ 駅: {len(df)} 駅\n■ 図を書き出します")

    fig_distribution(df)
    fig_correlation(df)

    for path, func, need in ((REG, fig_residual, "05"),
                             (CV, fig_validation, "08")):
        if not os.path.exists(path):
            print(f"  [!] {path} が無いので飛ばします（{need} を先に実行）")
            continue
        d = pd.read_csv(path)
        func(d) if path == CV else func(df, d)

    print(f"\n→ {OUTDIR}/ に出力しました")


if __name__ == "__main__":
    main()

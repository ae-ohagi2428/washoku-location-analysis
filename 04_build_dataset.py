"""
04. 分析用データセットをつくる
  駅・店舗・メッシュ統計・地価を1つのテーブルにまとめる。

やること:
  (1) 店舗を座標で最寄り駅に割り当てる
      ホットペッパーの最寄駅名だと、隣が大きな駅の場合、
      隣駅が最寄り駅だと判定されてしまい、その駅の店舗が0店になってしまう。
      →独自に座標を用いて最寄り駅を判定する。
  (2) メッシュ統計を排他割り当て（メッシュ中心から最寄り駅・上限1.5km）で集計する
  (3) 地価は駅から最寄りの1地点を採る（価格を持つ地点に限る）

  ※ 都県境付近のメッシュ・店舗について、県外駅のほうが近い場合は
     正しく県外駅に紐づける必要がある。
     →01が書き出した紐づけ対象駅（linked_stations.csv）を読み込み、
       紐づけ先の候補に加える（分析対象は東京都にある駅のみ）。

必要なファイル:
  tokyo_stations.csv / linked_stations.csv
  G004_main_tokyo.csv / landprice_points.csv
  tblT001142*（国勢調査250m 人口及び世帯）
  tblT001163*（経済センサス500m 産業中分類）

出力:
  station_dataset.csv     分析用テーブル（1駅につき1行）

実行:
  pip install pandas numpy
  python 04_build_dataset.py
"""
import glob, os
import numpy as np
import pandas as pd

STATIONS  = "data/tokyo_stations.csv"
LINKED_STATIONS = "data/linked_stations.csv"
WASHOKU   = "data/G004_main_tokyo.csv"
LANDPRICE = "data/landprice_points.csv"
OUT_CSV   = "data/station_dataset.csv"

CAP_KM = 1.5                 # メッシュ割り当ての距離上限

# 島しょ部の店は除外する（駅がないため、無理に本土の駅へ割り当てない）
EXCLUDE_ADDR = ["八丈町", "大島町", "三宅村", "御蔵島村", "青ヶ島村",
                "利島村", "新島村", "神津島村", "小笠原村"]

# 価格帯の区切り。コードの並びは安い順ではない点に注意。
BUDGET_GROUPS = {
    "casual": ["B009", "B010", "B011", "B001", "B002", "B003", "B008"],  # 〜5000円
    "middle": ["B015", "B016", "B017", "B018", "B019"],                  # 5001〜10000円
    "luxury": ["B020", "B021", "B012", "B013", "B014"],                  # 10001円〜
}

COLS_POP = {"T001142001": "pop"}
COLS_ECO = {"T001163082": "estab_food", "T001163081": "estab_hotel"}


def nearest(plat, plon, slat, slon, cap_km=None):
    """各点を最寄りの駅に割り当てる。戻り値は駅インデックス（上限超は -1）。"""
    idx = np.full(len(plat), -1, dtype=int)
    dst = np.full(len(plat), np.inf)
    CH = 3000
    for s in range(0, len(plat), CH):
        e = min(s + CH, len(plat))
        dy = (plat[s:e, None] - slat[None, :]) * 111.0
        dx = (plon[s:e, None] - slon[None, :]) * 111.0 * \
             np.cos(np.radians((plat[s:e, None] + slat[None, :]) / 2))
        d = np.hypot(dy, dx)
        k = d.argmin(axis=1)
        best = d[np.arange(e - s), k]
        idx[s:e] = k if cap_km is None else np.where(best <= cap_km, k, -1)
        dst[s:e] = best
    return idx, dst


def load_linked_stations():
    """01が書き出した紐づけ対象駅を読み込む。
    隣接県の駅と、東京都内で乗降客数が突合できなかった駅が入っている。
    分析対象には含めないが、店舗やメッシュの紐づけ先の候補として使う。"""
    try:
        df = pd.read_csv(LINKED_STATIONS)
    except FileNotFoundError:
        print(f"  [!] {LINKED_STATIONS} が無いので紐づけ対象駅なしで進めます")
        print("      01を実行して作成してください")
        return np.array([]), np.array([])
    print(f"  紐づけ対象駅: {len(df):,} 駅")
    print("    ※ 分析対象には含めない。"
          "店舗やメッシュがこちらに付いた場合は集計から外す")
    print("    ※ 駅の密集度（n_stations_1500m）のカウントには含める")
    return df["lat"].to_numpy(float), df["lon"].to_numpy(float)


def mesh_center(codes):
    """8桁(1km)/9桁(500m)/10桁(250m)のメッシュコードから中心の緯度経度を返す。"""
    s = codes.astype(str)
    lat = s.str[0:2].astype(float) / 1.5 + s.str[4:5].astype(float) / 12 \
        + s.str[6:7].astype(float) / 120
    lon = s.str[2:4].astype(float) + 100 + s.str[5:6].astype(float) / 8 \
        + s.str[7:8].astype(float) / 80
    dlat, dlon = np.full(len(s), 1/120), np.full(len(s), 1/80)
    n = s.str.len()
    for pos, need in ((8, 9), (9, 10)):     # 4次(500m) → 5次(250m)
        m = s.str[pos:pos+1].replace("", "0").astype(float).where(n >= need, 0)
        lat = lat + np.where(n >= need, ((m - 1) // 2).clip(0) * dlat / 2, 0)
        lon = lon + np.where(n >= need, ((m - 1) % 2).clip(0) * dlon / 2, 0)
        dlat = np.where(n >= need, dlat / 2, dlat)
        dlon = np.where(n >= need, dlon / 2, dlon)
    return lat + dlat / 2, lon + dlon / 2


def read_mesh(pattern, cols):
    hits = sorted(glob.glob(pattern))
    if not hits:
        print(f"  [!] {pattern} が見つかりません"); return None
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(hits[0], encoding=enc, dtype=str, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    # 1行目が項目コード、2行目が日本語の項目名。2行目は削除。
    if len(df) and not str(df.iloc[0].get("KEY_CODE", "")).strip().isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    keep = ["KEY_CODE"] + [c for c in cols if c in df.columns]
    df = df[keep].copy()
    for c in cols:
        if c in df.columns:
            # 「*」は秘匿。値は近隣メッシュに合算される仕様となっている
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["lat"], df["lon"] = mesh_center(df["KEY_CODE"])
    print(f"  {os.path.basename(hits[0])}: {len(df):,} メッシュ")
    return df.rename(columns=cols)


def main():
    st = pd.read_csv(STATIONS)
    print(f"■ 駅: {len(st)} 駅")

    slat, slon = st["lat"].to_numpy(), st["lon"].to_numpy()
    lk_lat, lk_lon = load_linked_stations()
    all_lat = np.concatenate([slat, lk_lat])
    all_lon = np.concatenate([slon, lk_lon])
    n_st = len(st)
    print()

    # ---- (1) 店舗を座標で最寄り駅へ ----
    shops = pd.read_csv(WASHOKU, low_memory=False)
    n0 = len(shops)
    drop = (shops["address"].fillna("").str.contains("|".join(EXCLUDE_ADDR))
            | (shops["station_name"].fillna("").astype(str).str.strip() == "")
            | shops["lat"].isna() | shops["lng"].isna())
    shops = shops[~drop].copy()
    idx, dist = nearest(shops["lat"].to_numpy(float),
                        shops["lng"].to_numpy(float), all_lat, all_lon)
    shops["_st"], shops["_dist_km"] = idx, dist
    wa = shops[shops["_st"] < n_st].copy()
    print(f"■ 和食（メインジャンル）: {n0:,} 件 → 除外 {n0-len(shops):,} / "
          f"紐づけ対象駅へ {len(shops)-len(wa):,} / 割り当て {len(wa):,} 件"
          f"（駅まで中央値 {wa['_dist_km'].median():.2f} km）")

    code2group = {c: g for g, cs in BUDGET_GROUPS.items() for c in cs}
    wa["bg"] = wa["budget_code"].map(code2group)

    st["shop_count"] = st.index.map(
        wa.groupby("_st").size()).fillna(0).astype(int)
    piv = (wa[wa["bg"].notna()]
           .pivot_table(index="_st", columns="bg", values="id", aggfunc="count")
           .fillna(0).astype(int))
    for g in BUDGET_GROUPS:
        if g not in piv.columns:
            piv[g] = 0
        st[g] = st.index.map(piv[g]).fillna(0).astype(int)
    print(f"    価格帯: " + " / ".join(
        f"{g} {int(st[g].sum()):,}" for g in BUDGET_GROUPS))
    n_nb = int(wa["bg"].isna().sum())
    if n_nb:
        print(f"    ※ 価格帯コードが無い店 {n_nb:,} 件は "
              f"shop_count のみに計上")
    print()

    # ---- (2) メッシュ統計 ----
    print("■ メッシュ統計")
    for pattern, cols in (("raw/tblT001142*", COLS_POP),
                          ("raw/tblT001163*", COLS_ECO)):
        mesh = read_mesh(pattern, cols)
        if mesh is None:
            continue
        idx, _ = nearest(mesh["lat"].to_numpy(float), mesh["lon"].to_numpy(float),
                         all_lat, all_lon, CAP_KM)
        m = mesh.assign(_st=idx)
        m = m[(m["_st"] >= 0) & (m["_st"] < n_st)]
        names = [v for v in cols.values() if v in m.columns]

        if names:
            a = m.groupby("_st")[names].sum(min_count=1)
            for n in names:
                st[f"{n}_1500m"] = st.index.map(a[n]).fillna(0).astype(int)
        print(f"    → {len(m):,} メッシュを割り当て")
    print()

    # ---- 駅の密集度（紐づけ対象駅も数える）----
    dy = (slat[:, None] - all_lat[None, :]) * 111.0
    dx = (slon[:, None] - all_lon[None, :]) * 111.0 * \
         np.cos(np.radians((slat[:, None] + all_lat[None, :]) / 2))
    st["n_stations_1500m"] = ((np.hypot(dy, dx) <= CAP_KM).sum(axis=1) - 1)

    # ---- (3) 地価：最寄りの1地点 ----
    pts = pd.read_csv(LANDPRICE)
    print("■ 地価")
    # 駅から見て最も近い地点を採る（地点→駅ではなく駅→地点の向き）
    plat, plon = pts["lat"].to_numpy(float), pts["lon"].to_numpy(float)
    dy = (slat[:, None] - plat[None, :]) * 111.0
    dx = (slon[:, None] - plon[None, :]) * 111.0 * \
         np.cos(np.radians((slat[:, None] + plat[None, :]) / 2))
    dd = np.hypot(dy, dx)
    near = dd.argmin(axis=1)
    st["land_price_commercial"] = pts["price"].to_numpy()[near]
    st["land_dist_commercial"] = dd[np.arange(len(st)), near].round(3)
    print(f"    距離の中央値 {st['land_dist_commercial'].median():.2f} km"
          f" / 2km超 {(st['land_dist_commercial']>2).sum()} 駅")

    st = st.drop(columns=["key"], errors="ignore")
    st.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_CSV}（{len(st)} 駅 × {len(st.columns)} 列）\n")

    print("== 価格帯別　和食店がゼロの駅 ==")
    for g in BUDGET_GROUPS:
        z = (st[g] == 0).sum()
        print(f"  {g:8} 合計 {int(st[g].sum()):>5} 店 / "
              f"ゼロ {z:>3} 駅（{z/len(st)*100:.1f}%）")

    print("\n== 和食店が多い駅 トップ10 ==")
    for _, r in st.nlargest(10, "shop_count").iterrows():
        print(f"  {r['station_name']:　<8} 計{int(r['shop_count']):>4}"
            f"（高級帯{int(r['luxury']):>3} 中間帯{int(r['middle']):>3}"
            f" 安価帯{int(r['casual']):>3}）乗降{int(r['passengers']):>9,}")


if __name__ == "__main__":
    main()

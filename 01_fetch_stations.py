"""
01. 東京の駅リストをつくる
  駅データ.jp の station20260731free.csv から東京都の駅を抽出し、
  国土数値情報 S12 の 2024年乗降客数を突合する。

  あわせて「紐づけ対象駅リスト」も書き出す。
  これは分析対象ではないが、店舗やメッシュを最寄り駅へ紐づける作業（04で行う）において
  参加させる駅で、次の2種類が入る。
    ・隣接県（埼玉・千葉・神奈川・山梨）の駅
        都県境において、都内の駅より県外駅が近いメッシュ・店舗を正しく紐づけるため。
    ・東京都内で乗降客数が突合できず、分析対象から外れた駅
        とうきょうスカイツリー、青梅線の御嶽・沢井など。
        周辺の別の駅にデータが紐づいてしまうのを防ぐため。

※ 駅名が同じ駅グループ（浅草の東武/メトロ/都営 と TX など）は1駅にまとめる。
   乗降客数が二重に計上されるのを避けるため。

必要なファイル（raw/ に置く）:
  station20260731free.csv           駅データ.jp の全国データ
  S12-*.geojson                     国土数値情報 駅別乗降客数（JGD2011・UTF-8版）

出力:
  tokyo_stations.csv                分析対象の駅リスト
  unmatched_stations.csv            乗降客数が突合できなかった駅
  linked_stations.csv           紐づけの対象として使う駅（04で読む）

実行:
  pip install pandas numpy
  python 01_fetch_stations.py
"""
import glob, json, math, re, sys, unicodedata
import numpy as np
import pandas as pd

STATION_CSV = "raw/station20260731free.csv"
OUT_CSV     = "data/tokyo_stations.csv"
UNMATCHED   = "data/unmatched_stations.csv"
LINKED_STATIONS  = "data/linked_stations.csv"

PREF_CD  = 13                        # 東京都（分析対象）
PREF_CDS = [11, 12, 13, 14, 19]      # 埼玉・千葉・東京・神奈川・山梨（データ紐づけの対象とする）

# 路面電車等（都電荒川線・東急世田谷線）は対象外とする。
# ただし乗り入れのある駅（三軒茶屋・大塚など）は他路線が残るので消えない。
EXCLUDE_LINE_CD = {99305, 26007}  # 都電荒川線 / 東急世田谷線

YEAR_VALUE  = "S12_061"           # 2024年の乗降客数
YEAR_FLAG   = "S12_059"           # 2024年のデータ有無コード（1=データ有）
MAX_DIST_KM = 2.0                 # 同名でもこれ以上離れていれば別の駅とみなす
DEDUP_KM    = 0.3                 # これより近ければ分析対象の駅と同一とみなす


def normalize(name):
    """駅名の表記ゆれを吸収する。
    NFKC で異体字・全角数字を統一し、括弧の中身と「駅」を落とす。
    例) 押上〈スカイツリー前〉→押上 / 笹塚(異体字)→笹塚 / 羽田空港第３→羽田空港第3
    """
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKC", name.strip())
    s = re.sub(r"[〈（(【\[].*?[〉）)】\]]", "", s)
    s = s.replace("ケ", "ヶ").replace("ガ", "ヶ").replace("が", "ヶ")
    s = re.sub(r"[\s　]", "", s)
    s = re.sub(r"駅$", "", s)
    return s


def dist_km(lat1, lon1, lat2, lon2):
    """2点間のおおよその距離(km)。緯度経度を平面近似して計算する。"""
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dlat, dlon)


def min_dist_km(plat, plon, slat, slon):
    """各点について、駅群までの最短距離(km)をまとめて求める。"""
    dy = (plat[:, None] - slat[None, :]) * 111.0
    dx = (plon[:, None] - slon[None, :]) * 111.0 * \
         np.cos(np.radians((plat[:, None] + slat[None, :]) / 2))
    return np.hypot(dy, dx).min(axis=1)


def read_raw():
    """駅データ.jp の全国データを読み、廃止駅と路面電車を除外する。"""
    df = pd.read_csv(STATION_CSV, dtype={"post": str})
    print(f"■ 駅データ.jp 全国: {len(df):,} 行")
    df = df[df["e_status"] == 0]
    df = df[~df["line_cd"].isin(EXCLUDE_LINE_CD)]
    print(f"  廃止駅・路面電車を除外: {len(df):,} 行")
    return df


def load_stations(raw):
    df = raw[raw["pref_cd"] == PREF_CD].copy()
    print(f"  東京都: {len(df):,} 行")

    # 同一駅の複数路線を1行にまとめる。
    # alias_names には小田急◯◯／京王◯◯のような別名も残す。
    df["line_count"] = df.groupby("station_g_cd")["line_cd"].transform("nunique")
    g = (df.sort_values("station_cd")
           .groupby("station_g_cd", as_index=False)
           .agg(station_name=("station_name", "first"),
                alias_names=("station_name", lambda x: "|".join(sorted(set(x)))),
                lat=("lat", "mean"), lon=("lon", "mean"),
                address=("address", "first"),
                line_count=("line_count", "first")))
    print(f"  駅グループに集約: {len(g):,} 駅")

    # 同名の駅グループを1駅にまとめる（距離は見ず、駅名の一致だけで判定）。
    # 東京都内では同名で離れた駅の組が無いことを別途検証済み。
    # 例) 浅草は東武/メトロ/都営 と つくばエクスプレスで座標が 600m ほど離れており、
    #     駅データ.jp では別グループになっている。
    #     このまま進めると、同じ乗降客数のレコードを両方が拾って二重計上になるため統合が必要。
    before = len(g)
    g["key"] = g["station_name"].map(normalize)
    g = (g.groupby("key", as_index=False)
           .agg(station_g_cd=("station_g_cd", "first"),
                station_name=("station_name", "first"),
                alias_names=("alias_names", lambda x: "|".join(
                    sorted({a for s in x for a in str(s).split("|")}))),
                lat=("lat", "mean"), lon=("lon", "mean"),
                address=("address", "first"),
                line_count=("line_count", "sum")))
    if before != len(g):
        print(f"  同名の駅グループを統合: {before} → {len(g):,} 駅")
    print()
    return g.drop(columns=["key"])


def load_passengers():
    hits = sorted(glob.glob("raw/S12*.geojson"))
    if not hits:
        print("[エラー] S12*.geojson が見つかりません。"); sys.exit(1)
    if len(hits) > 1:
        print(f"[!] S12 の geojson が {len(hits)} 個あります: {', '.join(hits)}")
        print(f"    先頭の {hits[0]} を使います")
    print(f"■ 乗降客数: {hits[0]}")

    with open(hits[0], encoding="utf-8") as f:
        gj = json.load(f)

    rows = []
    for feat in gj.get("features", []):
        p = feat.get("properties", {})
        if p.get(YEAR_FLAG) != 1:          # データ有(=1)のみ
            continue
        v = p.get(YEAR_VALUE) or 0
        if v <= 0:                          # 0は実質欠損として除外
            continue
        # geometry は線。構成する点の平均を駅の代表点とする。
        pts = [c for c in (feat.get("geometry") or {}).get("coordinates", [])
               if isinstance(c, (list, tuple)) and len(c) >= 2]
        if not pts:
            continue
        rows.append({"key": normalize(p.get("S12_001")),
                     "lat": sum(c[1] for c in pts) / len(pts),
                     "lon": sum(c[0] for c in pts) / len(pts),
                     "passengers": v})
    df = pd.DataFrame(rows)
    print(f"  データ有・値>0 のレコード: {len(df):,} 行\n")
    return df


def save_linked_stations(raw, ok):
    """データ紐づけの対象として使う、隣接県等の駅データを準備する。
    1都4県の駅データのうち、分析対象とする東京の駅の情報を削除する。
    残るのは隣接県の駅と、東京で乗降客数が突合できなかった駅となる。"""
    df = raw[raw["pref_cd"].isin(PREF_CDS)]
    g = (df.groupby("station_g_cd", as_index=False)
           .agg(station_name=("station_name", "first"),
                pref_cd=("pref_cd", "first"),
                lat=("lat", "mean"), lon=("lon", "mean")))

    d = min_dist_km(g["lat"].to_numpy(float), g["lon"].to_numpy(float),
                    ok["lat"].to_numpy(float), ok["lon"].to_numpy(float))
    cp = g[d > DEDUP_KM].copy()

    print("== 分析対象ではないが、データ紐づけの対象とする駅 ==")
    print(f"  1都4県 {len(g):,} 駅 → 分析対象と同じ駅を除いて {len(cp):,} 駅")
    cp.to_csv(LINKED_STATIONS, index=False, encoding="utf-8-sig")
    print(f"→ {LINKED_STATIONS} ({len(cp)}駅)\n")


def main():
    raw = read_raw()
    st = load_stations(raw)
    ksj = load_passengers()
    by_key = {k: g for k, g in ksj.groupby("key")}

    # 駅名が一致し、かつ座標が近いものだけを突合する。
    # 名前だけで突合すると全国の同名駅が合算される（例：東京の京橋に大阪の京橋）。
    res = []
    for _, s in st.iterrows():
        cand = by_key.get(normalize(s["station_name"]))
        total, n_rec = None, 0
        if cand is not None:
            d = cand.apply(lambda r: dist_km(s["lat"], s["lon"],
                                             r["lat"], r["lon"]), axis=1)
            near = cand[d <= MAX_DIST_KM]
            if len(near):
                total, n_rec = int(near["passengers"].sum()), len(near)
        res.append({"passengers": total, "n_records": n_rec})

    df = pd.concat([st.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    ok = df[df["passengers"].notna()].copy()
    ok["passengers"] = ok["passengers"].astype(int)
    ng = df[df["passengers"].isna()].copy()

    print("== 突合結果 ==")
    print(f"  乗降客数あり: {len(ok):,} 駅 / できず: {len(ng):,} 駅"
          f"（{len(ok)/len(df)*100:.1f}%）\n")

    cols = ["station_g_cd", "station_name", "alias_names", "lat", "lon",
            "address", "line_count", "passengers", "n_records"]
    ok[cols].sort_values("passengers", ascending=False)\
            .to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    ng[["station_g_cd", "station_name", "address"]]\
            .to_csv(UNMATCHED, index=False, encoding="utf-8-sig")
    print(f"→ {OUT_CSV} ({len(ok)}駅) / {UNMATCHED} ({len(ng)}駅)\n")

    save_linked_stations(raw, ok)

    print("== 乗降客数トップ10 ==")
    for _, r in ok.nlargest(10, "passengers").iterrows():
        print(f"  {r['station_name']:　<10} {int(r['passengers']):>9,} 人/日"
              f"  ({int(r['line_count'])}路線)")


if __name__ == "__main__":
    main()

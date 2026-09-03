"""
03. 地価公示・地価調査を取得する
  不動産情報ライブラリ XPT002 から地価地点（点データ）を取得する。

設計:
  年次    2025（地価公示=1月1日 と 地価調査=7月1日 の両方がそろう年。
                2026だと地価調査が未公表で件数が減る）
  ズーム  13（zは範囲を変えるだけで粒度は変わらないと検証済み。取得枚数を抑える）
  用途    05商業地
          地点は駅からの距離が中央値0.16km・2km超は6駅のみと確認済み。

出力:
  landprice_points.csv    取得した地価地点
  ※ このファイルが既にあれば、APIから取得せず再利用する。取り直すときは削除する。

実行:
  pip install requests pandas
  pip install python-dotenv
  python 03_fetch_landprice.py
"""
import math, os, sys, time
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("REINFOLIB_API_KEY", "")
URL     = "https://www.reinfolib.mlit.go.jp/ex-api/external/XPT002"

STATIONS = "data/tokyo_stations.csv"
PTS_CSV  = "data/landprice_points.csv"
EMPTY_CSV = "data/landprice_empty_tiles.csv"

YEAR  = 2025
ZOOM  = 13    # ズームレベル。13（大字）～15（詳細）から指定
CATEGORY_CODE  = "05"           # 商業地
CATEGORY_LABEL = "commercial"
SLEEP = 1.0


def latlon_to_tile(lat, lon, z):
    """緯度経度から XYZ 方式のタイル座標を求める（メルカトル図法）。"""
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y

def to_number(s):
    """「37,600,000(円/㎡)」のような文字列から数値を取り出す。"""
    if s is None:
        return np.nan
    t = "".join(ch for ch in str(s) if ch.isdigit() or ch in ".-")
    try:
        return float(t) if t not in ("", "-", ".") else np.nan
    except ValueError:
        return np.nan


def fetch(x, y, cat):
    params = {"response_format": "geojson", "z": ZOOM, "x": x, "y": y,
              "year": YEAR, "useCategoryCode": cat}
    for _ in range(3):
        try:
            r = requests.get(URL, params=params,
                             headers={"Ocp-Apim-Subscription-Key": API_KEY},
                             timeout=30)
            time.sleep(SLEEP)
            if r.status_code == 200:
                return r.json().get("features", []) or []
            print(f"    [警告] status={r.status_code} {r.text[:100]}")
            time.sleep(3)
        except requests.RequestException as e:
            print(f"    [警告] 通信エラー: {e}")
            time.sleep(3)
    return None

def main():
    st = pd.read_csv(STATIONS)
    print(f"■ 駅: {len(st)} 駅")

    if os.path.exists(PTS_CSV):
        pts = pd.read_csv(PTS_CSV)
        print(f"■ {PTS_CSV} を再利用（{len(pts):,} 地点）")
        print("   取り直すときはこのファイルを削除してください")
        return

    # 駅のタイル＋その周囲8枚を対象にする（タイル境界の外にある近い地点も拾うため）
    tiles = set()
    for lat, lon in zip(st["lat"], st["lon"]):
        tx, ty = latlon_to_tile(lat, lon, ZOOM)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tiles.add((tx + dx, ty + dy))
    tiles = sorted(tiles)
    print(f"■ 取得するタイル: {len(tiles)} 枚"
          f" = {len(tiles)} リクエスト"
          f"（約{len(tiles)*SLEEP/60:.0f}分）\n")

    rows, empty = [], []
    for i, (x, y) in enumerate(tiles, 1):
        feats = fetch(x, y, CATEGORY_CODE)
        if feats is None or not feats:
            empty.append((x, y, CATEGORY_LABEL))
        else:
            for f in feats:
                p = f.get("properties", {})
                c = f.get("geometry", {}).get("coordinates") or [None, None]
                rows.append({
                    "point_id": p.get("point_id"),
                    "category": CATEGORY_LABEL,
                    "lon": c[0], "lat": c[1],
                    "price": to_number(p.get("u_current_years_price_ja")),
                    "change_rate": to_number(p.get("year_on_year_change_rate")),
                    "location": p.get("location"),
                    "nearest_station": p.get("nearest_station_name_ja"),
                    "target_year": p.get("target_year_name_ja"),
                })
        if i % 20 == 0 or i == len(tiles):
            print(f"  {i}/{len(tiles)} タイル  地点 {len(rows):,} 件")

    pts = pd.DataFrame(rows).drop_duplicates(subset="point_id")
    pts = pts[pts["lat"].notna() & pts["price"].notna()]

    pts.to_csv(PTS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n→ {PTS_CSV} に {len(pts):,} 地点を保存")

    if empty:
        e = pd.DataFrame(empty, columns=["x", "y", "category"])
        e["z"] = ZOOM
        e.to_csv(EMPTY_CSV, index=False, encoding="utf-8-sig")
        print(f"\n[!] 地点が0件だったタイル: {len(empty)} 件"
              f" → {EMPTY_CSV} に出力")
        print("    ※ 山間部・河川敷など、地価地点が無いタイル")

if __name__ == "__main__":
    if not API_KEY:
        print("[エラー] API_KEY を設定してください。"); sys.exit(1)
    main()

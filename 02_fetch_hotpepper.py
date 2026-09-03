"""
02. ホットペッパーグルメ　グルメサーチAPIから東京の和食店（G004）を取得する

  APIの genre 指定はメインジャンル・サブジャンルの両方を検索対象にするため、
  G004 を指定するとサブジャンルだけが和食の店（居酒屋など）も返ってくる。
  そこで取得後にメイン／サブで分けて、2つのファイルに保存する。

  分析の目的変数に使うのはメインジャンルが和食の店のみ。
  サブのみの店は変数の作成には使用しないが、
  「除外の妥当性を確かめるための比較用」として残す。

出力:
  G004_main_tokyo.csv   メインジャンルが和食の店（分析に使う）
  G004_sub_tokyo.csv    サブジャンルのみ和食の店（比較用）

実行:
  pip install requests pandas
  pip install python-dotenv
  python 02_fetch_hotpepper.py
"""
import os, sys, time
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("HOTPEPPER_API_KEY", "")
URL     = "https://webservice.recruit.co.jp/hotpepper/gourmet/v1/"

AREA     = "Z011"     # 東京
GENRE    = "G004"     # 和食
PER_PAGE = 100        # 1リクエストの最大件数
SLEEP    = 1.0

OUT_MAIN = "data/G004_main_tokyo.csv"
OUT_SUB  = "data/G004_sub_tokyo.csv"

FIELDS = ["id", "name", "station_name", "address", "lat", "lng",
          "main_genre_code", "main_genre_name",
          "sub_genre_code", "sub_genre_name",
          "budget_code", "budget_name", "capacity",
          "middle_area", "small_area", "lunch", "midnight"]


def fetch_page(start):
    params = {"key": API_KEY, "large_area": AREA, "genre": GENRE,
              "count": PER_PAGE, "start": start, "format": "json"}
    for _ in range(3):
        try:
            r = requests.get(URL, params=params, timeout=30)
            time.sleep(SLEEP)
            if r.status_code == 200:
                return r.json().get("results", {})
            print(f"    [警告] status={r.status_code} {r.text[:120]}")
            time.sleep(3)
        except requests.RequestException as e:
            print(f"    [警告] 通信エラー: {e}")
            time.sleep(3)
    return None


def flatten(shop):
    g  = shop.get("genre") or {}
    sg = shop.get("sub_genre") or {}
    b  = shop.get("budget") or {}
    ma = shop.get("middle_area") or {}
    sa = shop.get("small_area") or {}
    return {"id": shop.get("id"), "name": shop.get("name"),
            "station_name": shop.get("station_name"),
            "address": shop.get("address"),
            "lat": shop.get("lat"), "lng": shop.get("lng"),
            "main_genre_code": g.get("code"), "main_genre_name": g.get("name"),
            "sub_genre_code": sg.get("code"), "sub_genre_name": sg.get("name"),
            "budget_code": b.get("code"), "budget_name": b.get("name"),
            "capacity": shop.get("capacity"),
            "middle_area": ma.get("name"), "small_area": sa.get("name"),
            "lunch": shop.get("lunch"), "midnight": shop.get("midnight")}


def main():
    os.makedirs(os.path.dirname(OUT_MAIN), exist_ok=True)
    print(f"{'='*58}\n■ 和食（{GENRE}）を取得\n{'='*58}")

    first = fetch_page(1)
    if not first:
        print("[エラー] 取得に失敗しました。"); return
    total = int(first.get("results_available", 0))
    n_req = -(-total // PER_PAGE)
    print(f"  対象: {total:,} 件 / {n_req} リクエスト"
          f"（約{n_req * SLEEP / 60:.0f}分）\n")

    rows = []
    for start in range(1, total + 1, PER_PAGE):
        res = first if start == 1 else fetch_page(start)
        got = res.get("shop", []) or []
        if not got:
            break
        rows.extend(flatten(s) for s in got)
        print(f"  {min(start + PER_PAGE - 1, total):>6,} / {total:,} 件")

    df = pd.DataFrame(rows, columns=FIELDS)
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    print(f"\n→ 取得 {len(df):,} 件")

    # ---- メイン／サブに分けて保存 ----
    is_main = df["main_genre_code"] == GENRE
    main, sub = df[is_main], df[~is_main]

    main.to_csv(OUT_MAIN, index=False, encoding="utf-8-sig")
    sub.to_csv(OUT_SUB, index=False, encoding="utf-8-sig")
    print(f"    {OUT_MAIN}  メインジャンルが和食 {len(main):,} 件")
    print(f"    {OUT_SUB}   サブジャンルのみ和食 {len(sub):,} 件")

    # ---- 除外の妥当性を確かめる比較（記事Ⅱ(2)の根拠）----
    print("\n== メイン／サブの営業形態の比較 ==")
    print(f"  {'':16}{'メイン':>10}{'サブのみ':>10}")
    print(f"  {'席数の中央値':16}{main['capacity'].median():>10.0f}"
          f"{sub['capacity'].median():>10.0f}")
    for label, col, val in (("深夜営業の割合", "midnight", "営業している"),
                            ("ランチありの割合", "lunch", "あり")):
        a = (main[col] == val).mean() * 100
        b = (sub[col] == val).mean() * 100
        print(f"  {label:16}{a:>9.1f}%{b:>9.1f}%")

if __name__ == "__main__":
    if not API_KEY:
        print("[エラー] API_KEY を設定してください。"); sys.exit(1)
    main()

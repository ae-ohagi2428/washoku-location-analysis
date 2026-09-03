# 東京567駅における和食店の出店余地分析

## 概要
東京都内に和食店を出店することを想定し、「駅ごとの和食店数を説明するモデル」を作成することで、「和食店の出店余地がある都内の駅」を提示することを目的とした分析。なお、詳細な分析内容は[Zenn記事](https://zenn.dev/ohagi_2428/articles/81d55ec231f370/)で示している。

## 使用データ
分析に使用したデータは下記のとおり。いずれのデータについても、集計または加工したうえで使用している。加工内容等の詳細は[Zenn記事](https://zenn.dev/ohagi_2428/articles/81d55ec231f370/)参照のこと。
1. 「駅データ 2026-07-31」：[駅データ.jp](https://ekidata.jp)（株式会社バリューアンドビジョン）- 2026年8月27日利用
2. 「駅別乗降客数 2024年度（令和6年度）版」：国土交通省「[国土数値情報ダウンロードサイト](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-S12-2024.html)」- 2026年8月27日利用
3. 「ホットペッパーグルメ グルメサーチAPI」：リクルート「[ホットペッパーグルメ Webサービス](http://webservice.recruit.co.jp/)」- 2026年9月2日利用
Powered by [ホットペッパーグルメ Webサービス](http://webservice.recruit.co.jp/)
4. 「XPT002. 地価公示・地価調査のポイント（点）API」：国土交通省「[不動産情報ライブラリ](https://www.reinfolib.mlit.go.jp/)」- 2026年9月2日利用
5. 総務省統計局「国勢調査（2020年）5次メッシュ（250mメッシュ）人口及び世帯（JGD2011）：M5339」：独立行政法人統計センター「[政府統計の総合窓口（e-Stat）](https://www.e-stat.go.jp/gis/statmap-search?page=1&type=1&toukeiCode=00200521&toukeiYear=2020&aggregateUnit=Q&serveyId=Q002005112020&statsId=T001142&datum=2000&prefCode=13)」- 2026年8月27日利用
6. 総務省統計局「経済センサス－活動調査（2021年）4次メッシュ（500mメッシュ）産業（中分類）別事業所数及び従業者数（JGD2011）：M5339」：独立行政法人統計センター「[政府統計の総合窓口（e-Stat）](https://www.e-stat.go.jp/gis/statmap-search?page=1&type=1&toukeiCode=00200553&toukeiYear=2021&aggregateUnit=H&serveyId=H002005112021&statsId=T001163&datum=2011&prefCode=13)」- 2026年8月27日利用

## 分析手順
1. 東京都内の567駅を対象に、目的変数を駅ごとの「和食店数」、説明変数を「乗降客数」・「駅の密集度」・「飲食店数」・「宿泊施設数」・「商業地地価」・「夜間人口」とする線形重回帰モデルを価格帯別に作成。
2. 分析結果（「分析結果」欄1項に記載）を踏まえて、二段階モデル（ハードルモデル）の第一段階として店舗数が「0」か「1以上」かを判定するロジスティック回帰を作成。
3. 二段階モデルの第二段階として、店舗数が「1以上」である駅のみにデータを限定し、線形重回帰モデルを再度作成。
4. 2及び3で作成したモデルに実データを当てはめ、予測値との残差・的中率から、それぞれの価格帯における出店候補駅を算出。
5. 1で作成したモデルについて、負の二項回帰モデルとの比較を行い、結果の差異を確認。
6. 1・2・3で作成したモデルについて、汎化性能を確認するために交差検証を実施。

## 分析結果
1. モデルを実データへ適用し、残差から出店候補駅を算出したが、実際の店舗数が「0」の駅で大半が埋まってしまう結果であった。
2. ロジスティック回帰モデル・線形重回帰モデルの二段階モデルを作成し、和食店出店候補駅について、候補A（実際の店舗数が「0」の駅の中で見込み（確率）の上位10駅）及び候補B（実際の店舗数が「1以上」の駅で残差のマイナス値の上位10駅）を価格帯ごとに示した。
3. 線形重回帰と負の二項回帰を比較した結果、安価帯・中間帯においては手法による差はほとんどなかった。一方高級帯では上位15駅のうち約半数で結果が一致しなかった。
4. 交差検証から、第二段階の高級帯以外では汎化性能があることを確認した。第二段階の高級帯においては、過学習の疑いが生じた。
5. 以上から、作成したモデルにはいくつかの限界があるものの、出店候補地を絞り込む一助としてある程度機能すると結論づけた。

## 使用技術
### 言語・環境
- Python 3.13.5

### ライブラリ
| ライブラリ | 用途 |
|---|---|
| pandas | データ加工・集計 |
| NumPy | 数値計算 |
| requests | APIからのデータ取得 |
| python-dotenv | 環境変数・APIキーの管理 |
| statsmodels | 回帰分析・統計的検定 |
| SciPy | 統計計算 |
| scikit-learn | 機械学習・データ分割 |
| Matplotlib | データ可視化 |
| japanize-matplotlib | Matplotlibの日本語表示 |

## ディレクトリ構成
```
washoku-location-analysis/
│
├── raw/              # 生データ配置用（同梱なし）
├── data/             # 中間データ（同梱なし）
├── output/           # 最終的な作成データ（同梱なし）
│
├── 01_fetch_stations.py
├── 02_fetch_hotpepper.py
├── 03_fetch_landprice.py
├── 04_build_dataset.py
├── 05_analyze.py
├── 06_hurdle.py
├── 07_compare_models.py
├── 08_cross_validate.py
├── 09_visualize.py
│
├── README.md         # 本ファイル
└── requirements.txt
```
※　同梱なしとしたデータの作成については、再現方法の実行手順を参照。

## 再現方法
### 動作環境
- Python 3.13.5
- ライブラリは `requirements.txt` を参照
```
pip install -r requirements.txt
```

### 事前準備
#### 1. 生データの配置
プロジェクト直下に `raw/` `data/` `output/` の3つのフォルダを作成する。下記の4ファイルをダウンロードして配置する。

| ファイル | 入手先 |
|---|---|
| `station20260731free.csv` | [駅データ.jp](https://ekidata.jp)（無料会員登録が必要） |
| `S12-25_NumberOfPassengers.geojson` | [国土数値情報ダウンロードサイト](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-S12-2024.html)<br>`S12-25_GML.zip` を展開し、UTF-8フォルダ内の `.geojson` を配置 |
| `tblT001142Q5339.txt` | [e-Stat 国勢調査 250mメッシュ](https://www.e-stat.go.jp/gis/statmap-search?page=1&type=1&toukeiCode=00200521&toukeiYear=2020&aggregateUnit=Q&serveyId=Q002005112020&statsId=T001142&datum=2000&prefCode=13)<br>zipを展開して配置 |
| `tblT001163H5339.txt` | [e-Stat 経済センサス 500mメッシュ](https://www.e-stat.go.jp/gis/statmap-search?page=1&type=1&toukeiCode=00200553&toukeiYear=2021&aggregateUnit=H&serveyId=H002005112021&statsId=T001163&datum=2011&prefCode=13)<br>zipを展開して配置 |

#### 2. APIキーの設定
プロジェクト直下に `.env` を作成し、下記2つのキーを記載する。
```
HOTPEPPER_API_KEY=取得したキー
REINFOLIB_API_KEY=取得したキー
```
- ホットペッパーグルメ グルメサーチAPI：[リクルートWEBサービス](https://webservice.recruit.co.jp/register/)で取得
- 地価公示・地価調査のポイント（点）API：[不動産情報ライブラリ](https://www.reinfolib.mlit.go.jp/api/request/)で利用申請

### 実行手順
01から09の順に実行する。
```
python 01_fetch_stations.py # 駅リストの作成
python 02_fetch_hotpepper.py # 和食店データの取得（約1分）
python 03_fetch_landprice.py # 地価データの取得（約3分）
python 04_build_dataset.py # 分析用データセットの作成
python 05_analyze.py # 線形重回帰
python 06_hurdle.py # 二段階モデル
python 07_compare_models.py # 負の二項回帰との比較
python 08_cross_validate.py # 交差検証
python 09_visualize.py # 図の書き出し
```
- 02・03はAPIを利用するため、実行時点のデータにより結果が変わる場合がある。
- 03は `data/landprice_points.csv` が存在する場合、再取得せずに再利用する。取り直す際はファイルを削除する。



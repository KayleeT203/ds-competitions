# 「成長企業」特定・分析プロジェクト
EDINET APIを使用し、上場企業の連結売上高を取得。そのデータを元にDuckDBで成長率40％以上の企業を特定・分析するプロジェクトです。

## Step 1
### 対象企業の範囲
特定日本企業９社を指定しました。自動車・電気・ゲームなど複数の業種を選定しています。

下記９社の企業：
|　企業名 |
| --- |
| トヨタ自動車株式会社 |
| ソニーグループ株式会社 |
| 任天堂株式会社 |
| 本田技研工業 |
| 三菱商事株式会社 |
| 日産自動車株式会社 |
| パナソニックホールディングス株式会社 |
| マツダ株式会社 |
| 株式会社バンダイナムコホールディングス |

### 対象期間
各社の2026年6月に提出された有価証券報告書を元に当期・前期・前々期・３期前・４期前（計５年分）の連結売上高を取得し、net_value_clients.csvにデータとして保存しました。

実質的な比較期間は、新型コロナウイルス緊急事態宣言真っ只中の2022年（４期前）から2026年（当期）の約4年間です。収束後の各社の成長率を特定します。

### 取得する財務指標

**連結売上高**を取得しています。ただし、対象企業には日本基準(JGAAP)企業と
IFRS(国際会計基準)企業が混在しており、それぞれXBRLタグの構造が異なるため、
以下の優先順位で値を抽出しています。

1. **経営指標等の推移系タグ**(5期分の推移が記載されており、日本基準・IFRSともに対応)
   - 要素IDの末尾が `SummaryOfBusinessResults` または `KeyFinancialData` で終わり、
     かつ `Sales` または `Revenue` を含むもの
   - 例(日本基準): `jpcrp_cor:NetSalesSummaryOfBusinessResults`
   - 例(IFRS・共通タグ): `jpcrp_cor:RevenueIFRSSummaryOfBusinessResults`
   - 例(IFRS・会社独自拡張タグ): `jpcrp030000-asr_E02144-000:OperatingRevenuesIFRSKeyFinancialData`(トヨタ)、
     `jpcrp030000-asr_E01777-000:SalesAndFinancialServicesRevenueIFRSKeyFinancialData`(ソニー)

2. **財務諸表本体のタグ**(フォールバック。当期・前期の2期分のみ)
   - `jppfs_cor:NetSales`(日本基準の企業のみ)

連結・個別の判定は「連結・個別」列ではなく、コンテキストIDが`CurrentYearDuration`
(または`PriorNYearDuration`)のみで、`NonConsolidatedMember`(非連結)が付いていないことで判定しています。
これは、EDINETの仕様上「連結・個別」列は`jppfs_cor:`系のタグにしか正しく反映されず、
それ以外(IFRS・経営指標要約タグなど)は一律「その他」と表示されるためです。

### 事前準備

1. 必要ライブラリをインストールしてください
​```bash
pip install -r requirements.txt
​```

2. `.env.example` を `.env` にリネームしてください
​```bash
cp .env.example .env
​```
（Google Drive経由でダウンロードした場合、`.env.example`が一覧に表示されないことがあります。その場合はお手数ですが `.env` という名前の新規ファイルを作成し、次のステップの内容を直接記載してください。

3. `.env` 内の `EDINET_API_KEY=` の後ろに、ご自身のEDINET APIキーを入力してください
​```
EDINET_API_KEY=your_actual_api_key_here
​```

### 実行方法

​```bash
python main.py
​```

### 出力ファイル

- `net_value_clients.csv`: 9社 × 5期分(当期〜4期前)の連結売上高一覧
  - 列: `company`(企業名), `doc_id`(EDINET書類番号), `relative_year`(相対年度: CurrentYear/Prior1Year〜Prior4Year), `net_sales_yen`(連結売上高、円)
- `(docID).zip`: EDINETからダウンロードした開示書類のCSVパッケージ(生データ、9社分)


### 注意点
EDINET上にて売上高（円）が記載されていない企業は除外します。


---

## Step2: DuckDBによる成長率算出

ステップ1で保存した`net_value_clients.csv`をDuckDBで読み込み、企業ごとの成長率を算出し、
**成長率40%以上の企業を特定**しました。

### 成長率の定義・計算式

各企業について、データセット内の**最も新しい期(当期)**と**最も古い期(4期前)**の
連結売上高を比較し、以下の式で算出しています。

```
成長率(%) = (当期の連結売上高 ÷ 4期前の連結売上高 − 1) × 100
```

- 単年度の増減ではなく、**約4年間の累計成長率**を採用しています。
- 中間の期(前期・前々期・3期前)は、今回採用した計算式(起点と終点の比較)には使用していませんが、
  データ抽出ロジックの検証や、将来的に年平均成長率(CAGR)などの別指標を算出する際の
  予備データとして`net_value_clients.csv`に保持しています。

### 実行方法(DuckDB SQL)

`net_value_clients.csv`は「1行 = 1社・1期分」の縦持ち(long format)のため、
DuckDBの`PIVOT`構文で「1行 = 1社」の横持ち(wide format)に変換してから、
同じ行内で当期と4期前の値を比較する形で成長率を算出しています。

```sql
COPY (
  WITH clients AS (
    PIVOT './net_value_clients.csv'
    ON relative_year
    USING first(net_sales_yen)
  ),
  clients_growth_rate AS (
    SELECT company,
           ROUND((((CurrentYear / Prior4Year) - 1) * 100), 2) AS growth_rate_2022to2026
    FROM clients
    WHERE growth_rate_2022to2026 >= 40
  )
  SELECT * FROM clients_growth_rate
) TO 'growth_rate_40.csv';
```

### 出力ファイル

- `growth_rate_40.csv`: 成長率40%以上と判定した企業の一覧
  - 列: `company`(企業名), `growth_rate_2022to2026`(成長率、%)

### 結果

9社中、以下5社が成長率40%以上と判定されました。

| 企業名 | 成長率(%) |
|---|---|
| マツダ | 57.62 |
| 本田技研工業 | 49.78 |
| 日産自動車 | 42.53 |
| トヨタ自動車 | 61.52 |
| バンダイナムコホールディングス | 51.61 |


---

## Step3: dbtプロジェクト化

ステップ2で行ったDuckDBでの変換・抽出ロジックをさらに堅牢にし、実務レベルのデータパイプラインとして管理するため、**dbt（dbt-duckdb）**を用いたプロジェクト化を行いました。

SQLのコンパイル、シードデータ管理、および成果物の外部CSV出力までをシームレスに自動化しています。

### dbtプロジェクトの基本構造

`dbt init` によって生成されたワークスペースから、主に以下のディレクトリ・ファイルを使用してパイプラインを構築しました。

```text
edinet_growth_analysis/
├── dbt_project.yml        # プロジェクト全体の設定ファイル
├── profiles.yml           # DuckDBデータベースへの接続設定（target='dev'）
├── seeds/
│   └── net_value_clients.csv  # ステップ1で取得した生データのCSV（インプット）
└── models/
    ├── schema.yml         # モデルの定義やデータ型の構成、テストを管理
    └── growth_rate_overEqual40.sql # 成長率算出 ＆ 40%以上抽出を行うメインロジック
```
### 実行手順
#### 1. 必要パッケージのインストール
pip install dbt-duckdb

#### 2. CSVファイルをDuckDBのデータベース内にテーブルとしてインポート
dbt seed

##### 3. パイプラインを実行し、モデル（SQL）の計算処理を回して成果物CSVを出力
dbt run

#### （運用の修正時など）古いキャッシュやコンパイル済みファイルを削除してクリーンにする場合
dbt clean

### 出力ファイル
output/growth_rate_overEqual40.csv

列構成: company, growth_rate_2022to2026（4年間のトータル成長率）。

## Step4: 可視化から読み取れる考察
#### Graph 1: growth_rate_from_2022to2026_all
上記のbar plotからHonda, Bandai, Nissan, Toyota, Mazdaの５社が4期前から成長率40％以上を達成したことが確認できます。中でも、Toyotaの成長率が最も高く60％超であり、Nissanが最も低いことが分かります。
この図から、この5社は4年前（新型コロナウイルス禍）と比較して大きく成長していることが読み取れます。

#### Graph 2: growth_rate_trend_2022to2026_all
上記のline plotでは、成長率40％以上を達成した5社の各評価期間における累計成長率の推移が確認できます。Graph 1とは対照的に、3期前以降は成長率が減少傾向にあります。3期前起点の成長率はすでに40％未満であり、2期前起点ではNissanが、1期前起点ではMazdaとNissanがマイナス成長となっていることが確認できます。

#### Graph 3: avg_growth_rate_trend_2022to2026
上記のbar plotから各評価期間における5社の平均成長率が確認できます。Graph 2と同様に、3期前起点から平均成長率が40％を下回り、1期前起点では平均で5％未満にとどまっています。

#### Graph 4: growth_rate_comparison_2022to2026
上記のbar plotから各評価期間における各社の成長率を比較できます。期間ごとの最高成長率は、4期前・3期前起点ではToyota、2期前・1期前起点ではBandaiとなっています。他社が成長率を大きく落とす中、Bandaiは各期間における成長率の低下幅が最も小さく、特に2期前起点では他社が10％前後にとどまる中、Bandaiのみ約30％の成長率を維持しています。

####　全Graphを経て
2022年（新型コロナウイルス禍）と比較すると、2026年時点では5社ともに明確な成長が確認できます。しかし、ウイルス収束後の3期前起点以降から成長率は鈍化しており、2期前起点以降はマイナス成長の企業も出始めています。その中で、Bandaiは成長率の低下はあるものの、3期前起点以降も他社平均を上回る水準を維持しています。

要因としては以下の3点が考えられます。
1. コロナ禍後の物価高上昇により、新車よりも中古車需要が増加した。

2. 自動車各社が日本生産・販売から海外生産・販売へシフトしており、円建ての売上高の成長率に影響している。

3. 平成レトロやガチャガチャ・一番くじの流行により、国内エンタメ消費が堅調に推移しており、これらに強みを持つBandaiが3期以降も安定した成長率を維持できている。

## Step5: まとめ
#### 所要時間
６日間

#### 各ステップで詰まった点と解決⽅法
Step 1ではAPIの使用方法の習得に時間を要しました。PythonでAPIを呼び起こしファイルを自動保存するという工程が初めてであったため、EDINET仕様書の精読・関連記事の参照。生成AIへの確認を組み合わせることで解決しました。

Step 2では複数企業のデータを1行1社の横持ち形式にまとめる処理に詰まりました。DuckDB公式ドキュメントのPIVOT構文を参照することで解決しました。

Step 3ではdbtの概念習得に時間を要しました。Youtube上のDuckDBのdbtオフィシャルチャンネルでの解説動画などを複数視聴した後、生成AIとの認識確認やコマンドの確認を行うことができました。

Step 4では成長率40％以上という絞り込み済みデータからどのように可視化が有効かを検討しました。過去のプロジェクトで作成した図やpandasの公式ドキュメントを参照しアイディアを得ました。

#### 成⻑率の定義の選択理由
成長率はn期前の売上高からどれだけ増減したかをパーセンテージで表します。使用した計算式は以下になります：
```text
（今期の売上高÷ ｎ期前の売上高 - 1）× 100
```
「今期÷ ｎ期前」でn期前を基準とした倍率を求め、そこから1を引くことで純粋な増加率を算出し、100％を掛けてパーセンテージに変換しています。

単年度の増減ではなく4年間の累積成長率を採用した理由は、コロナ禍（2022年）からの回復・成長をトータルで評価するためです。

#### dbt モデル構成の意図
Step 1で出力した連結売上高データ（net_value_clients.csv）をseedsにinputとして配置し、SQLモデルで成長率算出とフィルタリングを自動実行する構成にしました。これにより、対象企業や期間を変更した場合でも、net_value_clients.csvを差し替えてdbt seed・dbt runを実行するだけで結果が再現できます。
{{ ref('net_value_clients') }}でseedsのファイルを参照し、materialized='external'でCSVとして外部出力する設定にしています。

#### コードの品質向上のために⼯夫した点
わかりやすい変数名・適切なコメントの記載により、コードの可読性を高めました。エラーハンドリングとして、APIレスポンスのステータスコード確認・ZIPファイル内の対象ファイル存在確認・float変換時のtry/except処理を実装し、想定外のデータ形式にも柔軟に対応できる構造にしました。また、APIキーなどの機密情報はコード内に直接記載せず、`.env`ファイルと`python-dotenv`を用いて環境変数として管理し、`.gitignore`でリポジトリから除外することでセキュリティに配慮しました。

#### パイプライン全体を複数回実⾏した場合の挙動
1. main.pyを実行するとEDINETから再度データを取得し、net_value_clients.csvを上書きします。同じ企業・期間であれば同じデータが出力されます。
2. net_value_clients.csvをseedsに配置しdbt seedを実行すると、DuckDB内のテーブルが再作成されます。
3. dbt runを実行すると成長率の再計算が行われ、growth_rate_overEqual40.csvが上書きされます。

このパイプラインにより、同じinputからは同じoutputが得られます。

#### 各ファイルの役割と実⾏⼿順
下記の順番が実行手順になります。
1. requirements.txt
main.py の実行に必要な外部ライブラリ(requests, pandas, python-dotenv)が
記載されています。事前準備セクションの手順に従い、
`pip install -r requirements.txt` を実行してください。

2. main.py
このファイルには特定企業の連結売上高を一覧表にするpythonコードが入っています。APIキーはコード内に直接記載せず、`.env` ファイルから `python-dotenv` を使って読み込む構成にしています(詳細は「事前準備」を参照)。また、中盤には特定企業の名前リストや期間指定をしているコードがあり、希望の企業・期間に変更することができます。
このpythonファイルを実行することにより、ファイル内に期間内に有価証券報告書を提出した、企業の連結売上高を四期前から今期までをnet_value_clients.csvとして出力します。

3. net_value_clients.csvにて出力された特定企業の連結売上高一覧表が確認できます。

4. edinet_growth_analysis
dbtプロジェクト構築の際に作成されたファイルです。

5. edinet_growth_analysis/seeds
先ほど作成したnet_value_clients.csvをこのファイル上に入れます。seedsはinputを入れるファイルの役割です。コマンドのdbt seed, dbt runをpowershellやターミナル上に書くことにより、成長率４０％以上の企業が出力されます。

6. edinet_growth_analysis/output
成長率４０％以上の企業が出力されたファイルがこのファイルに保存されます。

7. growth_rate_overEqual40.csv
成長率４０％以上の企業が出力されたことによってできたファイルです。

8. growth_analysis_graph
このファイルにはStep4で可視化された図のスクリーンショットが保存されています。

#### AIの使用について
生成AIには主に以下の役割を任せました。
1. API, dbtの情報と認識確認。
2. 複雑な正規表現など自分では書けない部分のコード生成
3. dbtやenvの設定方法
4. エラーの原因特定と修正方法
5. README言葉の修正

自分で判断・実施したことは以下になります。
1. 生成コードが自分のプロジェクトに合うか取捨選択
2. エラーハンドリング
3. 対象企業・期間・分析テーマの選定と考察
4. 図の可視化の方向性
5. 公式ドキュメントや関連記事の参照






---

## 参考にした情報源

- [Request API data using Python in 8 minutes!](https://www.youtube.com/watch?v=JVQNywo4AbU) (YouTube)
- [【EDINET API活用】複数企業財務データを一括取得〜初心者でも簡単〜年度別連結経営指標篇｜レベル０投資家C.B.](https://note.com/bolian/n/n495ccb46d979)
- [EDINET APIの分類コードを整理する｜イワシ銀行](https://note.com/python_beginner/n/n4c1bb83bee83)
- [How to Zip a File and Extract from a Zip File in Python - Python ZipFile Module](https://www.youtube.com/watch?v=6njkk0I1uvQ) (YouTube)
- [How to Create A Csv File Using Python - GeeksforGeeks](https://www.geeksforgeeks.org/python/how-to-create-a-csv-file-using-python/)
- [PythonでEDINETのXBRLを分析する その8「XBRLインスタンスファイルから値を取得する」 - プログラミングと株式投資のブログ](https://www.quwechan.com/entry/2023/02/25/071117)
- [トヨタの売上高だけ5年分あるはずなのに2年分しか表示されない｜村正＠AI Director](https://note.com/joyous_phlox662/n/n3eb441fbfa62)
- [PIVOT Statement – DuckDB(公式ドキュメント)](https://duckdb.org/docs/current/sql/statements/pivot)
- [Add Seeds to you DAG | dbt Developer Hub](https://docs.getdbt.com/docs/build/seeds?version=2.0&name=Fusion)
- [Chart visualisation - pandas 3.0.4 documentation](https://pandas.pydata.org/docs/user_guide/visualization.html)
- [pandas.DataFrame.plot.bar - pandas 3.0.4 documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.bar.html#pandas.DataFrame.plot.bar)

## 注意事項

- EDINET APIキーは `.env` ファイルで管理し、`python-dotenv` を用いてコード内から読み込む方式にしています。`.env` は `.gitignore` に追加し、リポジトリには含めていません。キー名のみを記載した `.env.example` を代わりに同梱していますので、実行時はこれを `.env` にリネームしてご自身のキーを設定してください。
- 本ツールはXBRLタグのパターンマッチングで売上高を抽出しているため、対象企業を追加した場合、本ツールが対応していないタグ名(会社独自拡張タグなど)を使用している企業については、抽出ロジックの個別対応が必要になることがあります。
- net_value_clients.csv および growth_rate_overEqual40.csv はUTF-8で保存しています。Excelで直接ダブルクリックして開くと、会社名など日本語部分が文字化けする場合があります。
正しく表示するには、Excelの「データ」タブ→「テキストまたはCSVから」（From Text/CSV）を選択し、文字コードとして「UTF-8」を指定して開いてください。
- EDINETからダウンロードした生ZIPファイルは容量の都合上同梱していません。main.pyを実行することで再取得可能です。

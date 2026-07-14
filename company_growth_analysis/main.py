import requests
import datetime
import time
import zipfile
import pandas as pd
from dotenv import load_dotenv
import os


base_url = 'https://api.edinet-fsa.go.jp/api/v2/documents.json?'
load_dotenv()
api_key = os.environ.get("EDINET_API_KEY")

def get_client_info(date_str, doc_type=2):
    url = f"{base_url}date={date_str}&type={doc_type}&Subscription-Key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

target_companies = [
    "トヨタ自動車",
    "ソニーグループ",
    "任天堂",
    "本田技研工業",
    "三菱商事",
    "日産自動車",
    "パナソニックホールディングス",
    "マツダ",
    "バンダイナムコホールディングス"
]

start = datetime.date(2026, 6, 1)
end = datetime.date(2026, 6, 30)
day = start

target_docs = []  # (企業名, docID)

while day <= end:
    date_str = day.strftime("%Y-%m-%d")
    data = get_client_info(date_str)
    if data is None:
        day += datetime.timedelta(days=1)
        continue
    for doc in data["results"]:
        if doc.get("formCode") == "030000" and doc.get("docTypeCode") == "120" and doc.get("ordinanceCode") == "010":
            filer_name = doc.get("filerName", "")
            for company in target_companies:
                if company in filer_name:
                    print(date_str, doc["docID"], filer_name)
                    target_docs.append((company, doc["docID"]))
                    break
    day += datetime.timedelta(days=1)

def download_csv(doc_id, api_key):
    url = f"https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}?type=5&Subscription-Key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    with open(f"{doc_id}.zip", "wb") as f:
        f.write(response.content)
    print(f"{doc_id}.zip として保存しました")
    return True

def extract_consolidated_sales(doc_id, year_difference=0):
    zip_name = f'{doc_id}.zip'
    with zipfile.ZipFile(zip_name) as z:
        target_file = None
        for name in z.namelist():
            if "jpcrp030000-asr" in name and name.endswith(".csv"):
                target_file = name
                break
        if target_file is None:
            return None
        with z.open(target_file) as f:
            df = pd.read_csv(f, sep="\t", encoding="utf-16")

    if year_difference == 0:
        current = df[df["コンテキストID"] == "CurrentYearDuration"]
    elif year_difference == 1:
        current = df[df["コンテキストID"] == "Prior1YearDuration"]
    elif year_difference == 2:
        current = df[df["コンテキストID"] == "Prior2YearDuration"]
    elif year_difference == 3:
        current = df[df["コンテキストID"] == "Prior3YearDuration"]
    else: # prior 4 year
        current = df[df["コンテキストID"] == "Prior4YearDuration"]

    # 日本基準（JGAAP）
    row = current[current["要素ID"] == "jppfs_cor:NetSales"]
    if not row.empty:
        try:
            return float(str(row.iloc[0]["値"]).replace(",", ""))
        except ValueError:
            return None

    # IFRS（国際会計基準）
    ifrs_row = current[
        current["要素ID"].astype(str).str.contains(r"(?:SummaryOfBusinessResults|KeyFinancialData)$", regex=True, na=False
        )
        & current["要素ID"].astype(str).str.contains("Sales|Revenue", regex=True, na=False)
    ]
    if not ifrs_row.empty:
        try:
            return float(str(ifrs_row.iloc[0]["値"]).replace(",", ""))
        except ValueError:
            return None

    return None

csv_file_path = './net_value_clients.csv'
consolidated_sales = [] # company, doc_id, relative_year, and net_sales_yen
print(not len(target_docs))

less_value_companies = []

if not len(target_docs):
    print("No valid company found")
else:
    for company, doc_id in target_docs:
        # download zip files
        print(f"{company} をダウンロード中。。。")
        result = download_csv(doc_id, api_key)
        if result is None:
            print(f'{company} のダウンロードに失敗しました')
            continue

        # extract and add net sales info
        company_sales = []
        skip_company = False
        for num in range(5):
            year = 'CurrentYear' if num == 0 else f'Prior{num}Year'
            net_sales = extract_consolidated_sales(doc_id, num)
            if net_sales is None:
                print(f'{company}の{year}のデータが取得できなかったため、{company}を除外します')
                skip_company = True
                break
            company_sales.append({
            "company": company,
            "doc_id": doc_id,
            "relative_year": year,
            "net_sales_yen": net_sales,
            })
        if not skip_company:
            consolidated_sales.extend(company_sales)
        else:
            less_value_companies.append(company)
        time.sleep(1)

unique_companies = set(one_consol['company'] for one_consol in consolidated_sales)
print(f'最終件数は{len(unique_companies)}です')
print(f'除外された企業：{less_value_companies}')

unique_target_companies = set(company for company, doc_id in target_docs)
total = len(unique_companies) + len(less_value_companies)
if total == len(unique_target_companies):
    print('安全に連結売上高企業を抽出できました')
else:
    print(f'特定企業数：{len(unique_target_companies)}に対し、件数差分が{total}あります')

# save as csv
if len(consolidated_sales):
    df_result = pd.DataFrame(consolidated_sales)
    df_result["net_sales_yen"] = df_result["net_sales_yen"].astype(int)
    df_result.to_csv(csv_file_path, index=False, encoding="utf-8", sep=",")
    print(f"{csv_file_path} に保存しました")
    print(df_result)
else:
    print('連結売上高の一覧表が作成できませんでした')
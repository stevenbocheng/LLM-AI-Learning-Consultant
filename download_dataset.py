import kagglehub
import pandas as pd
import os
import glob

# ── 1. 下載資料集 ────────────────────────────────────────────
print("正在下載 Goodreads 資料集（約 2GB，請耐心等候）...")
path = kagglehub.dataset_download("ishanrealstate/goodreads-cleaned-dataset")
print(f"下載完成，路徑：{path}")

# ── 2. 找到 Parquet 檔案 ─────────────────────────────────────
parquet_files = glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
all_files = parquet_files + csv_files

print(f"\n找到的檔案：")
for f in all_files:
    print(f"  {f}")

if not all_files:
    raise FileNotFoundError("找不到資料檔案，請確認下載是否完整")

# ── 3. 讀取資料 ──────────────────────────────────────────────
target_file = all_files[0]
print(f"\n讀取：{target_file}")

if target_file.endswith(".parquet"):
    df = pd.read_parquet(target_file)
else:
    df = pd.read_csv(target_file, low_memory=False)

print(f"\n原始筆數：{len(df):,}")
print(f"欄位：{df.columns.tolist()}")
print(f"\n前兩筆預覽：")
print(df.head(2).to_string())

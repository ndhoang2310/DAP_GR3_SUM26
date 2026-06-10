"""
dvc_pull_and_merge.py - Automates fetching dataset contributions from DVC and merging them.

Usage:
    python src/data_collection/dvc_pull_and_merge.py
"""

import os
import sys
import subprocess
import shutil

def run_command(cmd, shell=True, check=True):
    print(f"\n> Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, text=True)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}: {cmd}")
        sys.exit(result.returncode)
    return result

def main():
    print("=" * 60)
    print("  DVC PULL & MERGE TOOL (DÀNH CHO LEADER)")
    print("=" * 60)

    # Ensure git fetch is run to see latest branches
    print("\n[INFO] Đang cập nhật danh sách các nhánh từ server...")
    run_command("git fetch --all", check=False)

    # 1. Ask which members to pull
    print("\nNhập danh sách mã thành viên muốn gộp dữ liệu (cách nhau bởi dấu phẩy).")
    print("Ví dụ: M01, M02, M03")
    members_input = input("Danh sách thành viên: ").strip()
    
    if not members_input:
        print("[ERROR] Danh sách không được để trống.")
        sys.exit(1)
        
    members = [m.strip().upper() for m in members_input.split(",") if m.strip()]
    
    contributions_dir = os.path.join("data", "contributions")
    if os.path.exists(contributions_dir):
        print(f"\n[INFO] Dọn dẹp thư mục {contributions_dir} cũ...")
        shutil.rmtree(contributions_dir, ignore_errors=True)
    
    os.makedirs(contributions_dir, exist_ok=True)

    # 2. Pull data using dvc get
    for member in members:
        branch_name = f"origin/data/{member}"
        member_dir = os.path.join(contributions_dir, member)
        dataset_out = os.path.join(member_dir, "dataset")
        
        print(f"\n[INFO] Đang kéo dữ liệu (dataset) của {member} từ DVC (branch: {branch_name})...")
        os.makedirs(member_dir, exist_ok=True)
        
        # dvc get . dataset --rev origin/data/M01 -o data/contributions/M01/dataset
        # The '.' means current local repo. DVC will clone it temp, checkout origin/data/M01, and pull from remote.
        cmd = f"dvc get . dataset --rev {branch_name} -o {dataset_out}"
        try:
            run_command(cmd)
            print(f"  [OK] Đã tải xong dataset của {member}")
        except SystemExit:
            print(f"  [ERROR] Không thể lấy dữ liệu của {member}. Hãy kiểm tra xem branch '{branch_name}' đã tồn tại trên Github chưa.")

    # 3. Call existing merge_datasets.py
    print("\n[INFO] Bắt đầu gộp dữ liệu bằng merge_datasets.py...")
    merge_script = os.path.join("src", "data_collection", "merge_datasets.py")
    if os.path.exists(merge_script):
        run_command(f"python {merge_script} --input {contributions_dir} --output dataset_master", check=False)
    else:
        print(f"[ERROR] Không tìm thấy {merge_script}")

    print("\n" + "=" * 60)
    print(" [DONE] Hoàn tất quá trình kéo và gộp dữ liệu Master!")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
dvc_sync.py - Automates tracking and pushing dataset contributions to DVC and Git.

Usage:
    python src/data_collection/dvc_sync.py
"""

import os
import sys
import subprocess

def run_command(cmd, shell=True, check=True):
    print(f"\n> Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, text=True)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}: {cmd}")
        sys.exit(result.returncode)
    return result

def main():
    print("=" * 60)
    print("  DVC DATA SYNC TOOL - UPLOAD TO GOOGLE DRIVE")
    print("=" * 60)

    # 1. Check if git and dvc exist
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        subprocess.run(["dvc", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("[ERROR] Không tìm thấy 'git' hoặc 'dvc'. Hãy đảm bảo đã cài đặt đầy đủ.")
        sys.exit(1)

    # Fetch latest branches from remote to avoid conflicts
    print("\n[INFO] Cập nhật thông tin từ server...")
    run_command("git fetch --all", check=False)

    # 2. Ask for Member ID
    member_id = input("Nhập mã thành viên của bạn (VD: M01, M02): ").strip().upper()
    if not member_id:
        print("[ERROR] Mã thành viên không được để trống.")
        sys.exit(1)
    
    branch_name = f"data/{member_id}"
    
    # 3. Create or checkout Git branch safely
    print(f"\n[INFO] Chuyển sang branch {branch_name}...")
    try:
        # Check if local branch exists
        res_local = subprocess.run(f"git show-ref --verify refs/heads/{branch_name}", shell=True, capture_output=True)
        # Check if remote branch exists
        res_remote = subprocess.run(f"git show-ref --verify refs/remotes/origin/{branch_name}", shell=True, capture_output=True)
        
        if res_local.returncode == 0:
            # Branch exists locally
            run_command(f"git checkout {branch_name}")
            if res_remote.returncode == 0:
                print("  [INFO] Cập nhật dữ liệu mới nhất từ server cho nhánh của bạn...")
                run_command(f"git pull origin {branch_name}", check=False)
        elif res_remote.returncode == 0:
            # Branch exists on remote but not locally
            run_command(f"git checkout -b {branch_name} origin/{branch_name}")
        else:
            # Branch does not exist anywhere
            run_command(f"git checkout -b {branch_name}")
    except Exception as e:
        print(f"[ERROR] Không thể checkout branch {branch_name}: {e}")
        sys.exit(1)

    # 4. Add data to DVC
    print("\n[INFO] Thêm dữ liệu vào DVC (Tracking)...")
    if os.path.exists(os.path.join("data", "raw_videos")):
        run_command("dvc add data/raw_videos", check=False)
    else:
        print("  [WARNING] Không tìm thấy thư mục data/raw_videos")

    if os.path.exists("dataset"):
        run_command("dvc add dataset", check=False)
    else:
        print("  [WARNING] Không tìm thấy thư mục dataset")

    # 5. Commit to Git
    print("\n[INFO] Lưu các file cấu hình DVC vào Git...")
    # Thêm từng file nếu tồn tại
    for f in ["data/raw_videos.dvc", "dataset.dvc", ".gitignore", "data/.gitignore"]:
        if os.path.exists(f):
            run_command(f"git add {f}", check=False)
    
    # Allow commit to fail if there's nothing to commit
    res = subprocess.run(f'git commit -m "Update dataset for {member_id}"', shell=True)
    if res.returncode == 0:
        print("  [OK] Đã commit vào Git.")
    else:
        print("  [INFO] Không có sự thay đổi dữ liệu mới nào để commit.")

    # 6. Push data to DVC (Google Drive)
    print("\n[INFO] Đang tải dữ liệu thực tế lên Google Drive qua DVC...")
    print("  *Lưu ý*: Lần đầu tiên chạy có thể DVC sẽ yêu cầu xác thực Google. Hãy làm theo link trên màn hình.")
    res = subprocess.run("dvc push", shell=True)
    if res.returncode != 0:
         print("\n[ERROR] DVC push thất bại! Vui lòng kiểm tra lại kết nối mạng hoặc xác thực Google Drive.")
         sys.exit(1)

    # 7. Push branch to Github
    print("\n[INFO] Đang đẩy thông tin phiên bản lên Github...")
    res = subprocess.run(f"git push -u origin {branch_name}", shell=True)
    if res.returncode != 0:
         print("\n[ERROR] Đẩy lên Github thất bại. Vui lòng kiểm tra lại quyền truy cập Github.")
         sys.exit(1)

    print("=" * 60)
    print(f" [DONE] Dữ liệu của bạn ({member_id}) đã được đồng bộ lên Google Drive và Github thành công!")
    print("=" * 60)
    print("\nHãy báo cho Leader (hoặc tạo Pull Request) để gộp dữ liệu.\n")

if __name__ == "__main__":
    main()

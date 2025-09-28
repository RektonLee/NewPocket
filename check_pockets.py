#!/usr/bin/env python3
import os
import re
import glob

# 统计所有目录和pocket文件
sample_dir = "/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples/"

# 1. 统计所有_10A.pdb文件
all_pocket_files = glob.glob(os.path.join(sample_dir, "*", "*_10A.pdb"))
print(f"总pocket文件数量: {len(all_pocket_files)}")

# 2. 统计kcat_6位数字格式的目录
kcat_pattern = os.path.join(sample_dir, "kcat_[0-9][0-9][0-9][0-9][0-9][0-9]")
kcat_dirs = glob.glob(kcat_pattern)
print(f"kcat_6位数字格式的目录: {len(kcat_dirs)}")

# 3. 统计其他格式的目录
all_dirs = glob.glob(os.path.join(sample_dir, "*"))
other_dirs = [d for d in all_dirs if os.path.isdir(d) and not re.match(r'.*kcat_\d{6}$', d)]
print(f"其他格式的目录: {len(other_dirs)}")

# 4. 统计kcat_6位数字目录中有pocket文件的数量
has_pocket = 0
no_pocket = 0

for dir_path in kcat_dirs:
    pocket_files = glob.glob(os.path.join(dir_path, "*_10A.pdb"))
    if pocket_files:
        has_pocket += 1
    else:
        no_pocket += 1

print(f"\nkcat_6位数字目录中:")
print(f"  有pocket文件的目录: {has_pocket}")
print(f"  没有pocket文件的目录: {no_pocket}")
print(f"  覆盖率: {has_pocket/len(kcat_dirs)*100:.1f}%")

# 5. 统计其他目录中的pocket文件
other_has_pocket = 0
for dir_path in other_dirs:
    pocket_files = glob.glob(os.path.join(dir_path, "*_10A.pdb"))
    if pocket_files:
        other_has_pocket += 1

print(f"\n其他格式目录中:")
print(f"  有pocket文件的目录: {other_has_pocket}")

# 6. 显示目录格式分布
print(f"\n目录格式分布:")
print(f"  kcat_6位数字: {len(kcat_dirs)}")
print(f"  其他格式: {len(other_dirs)}")
print(f"  总计: {len(kcat_dirs) + len(other_dirs)}")

# 7. 显示前5个其他格式的目录示例
print(f"\n其他格式目录示例:")
for i, dir_path in enumerate(other_dirs[:5]):
    dir_name = os.path.basename(dir_path)
    pocket_files = glob.glob(os.path.join(dir_path, "*_10A.pdb"))
    pocket_count = len(pocket_files)
    print(f"  {dir_name} -> {pocket_count}个pocket文件")

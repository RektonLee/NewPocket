import re
import os
from collections import defaultdict

def analyze_time_distribution(log_file_path):
    """
    分析日志文件中"耗时{time}秒"格式的时间分布，包括大于30秒的统计
    """
    # 正则表达式匹配"耗时{time}秒"，支持整数和小数
    pattern = r'耗时: (\d+\.?\d*)秒'
    
    # 存储时间分布
    time_counts = defaultdict(int)
    # 统计大于30秒的次数
    over_30_count = 0
    # 总记录数
    total_entries = 0

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                matches = re.findall(pattern, line)
                for match in matches:
                    try:
                        time_val = float(match)
                        time_counts[time_val] += 1
                        total_entries += 1
                        if time_val > 60:
                            over_30_count += 1
                    except ValueError:
                        print(f"警告：无法将 '{match}' 转换为数字，已跳过")

        # 输出统计结果
        print("=" * 40)
        print("时间分布统计结果：")
        print("=" * 40)
        print(f"总记录数: {total_entries} 条")
        print(f"耗时大于30秒的记录数: {over_30_count} 条")
        if total_entries > 0:
            print(f"占比: {over_30_count/total_entries*100:.2f}%")
        print("-" * 40)
        print("时间(秒) | 出现次数")
        print("-" * 20)
        
        # 按时间值排序输出
        for time_val in sorted(time_counts.keys()):
            print(f"{time_val:>8.2f} | {time_counts[time_val]:>6d}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{log_file_path}'")
    except Exception as e:
        print(f"处理文件时发生错误：{str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python time_analyzer.py <日志文件路径>")
        print("示例: python time_analyzer.py app.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    analyze_time_distribution(log_file)
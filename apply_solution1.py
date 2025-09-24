#!/usr/bin/env python3
"""
应用方案1的修改：跳过失败的样本，不使用dummy PDB
"""

import re

def apply_solution1():
    """应用方案1的修改到generate_pdb.py"""
    
    # 读取原文件
    with open('generate_pdb.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修改1: 更新_predict_single_batch函数中的PDB转换结果检查
    old_check = '''            # 检查PDB转换结果
            if len(pdb_contents) == 0:
                logging.warning("PDB转换结果为空，使用dummy PDB")
                return [(seq, make_dummy_pdb(seq)) for seq in sequences]
            
            if len(pdb_contents) != len(sequences):
                logging.warning(f"PDB数量({len(pdb_contents)})与序列数量({len(sequences)})不匹配")
                # 补齐缺失的PDB
                while len(pdb_contents) < len(sequences):
                    pdb_contents.append(make_dummy_pdb(sequences[len(pdb_contents)]))
            
            return [(seq, pdb) for seq, pdb in zip(sequences, pdb_contents)]'''
    
    new_check = '''            # 方案1：检查PDB转换结果，如果失败则抛出异常
            if len(pdb_contents) == 0:
                logging.error("PDB转换结果为空，跳过这些序列")
                raise ValueError("ESMFold输出为空，无法生成有效的PDB结构")
            
            if len(pdb_contents) != len(sequences):
                logging.error(f"PDB数量({len(pdb_contents)})与序列数量({len(sequences)})不匹配")
                raise ValueError(f"PDB转换结果数量不匹配: 期望{len(sequences)}个，实际{len(pdb_contents)}个")
            
            return [(seq, pdb) for seq, pdb in zip(sequences, pdb_contents)]'''
    
    content = content.replace(old_check, new_check)
    
    # 修改2: 更新predict_batch函数中的错误处理
    old_error_handling = '''                    except Exception as e2:
                        logging.error(f"Individual prediction failed for seq len={len(seq)}: {e2}")
                        results.append((idx, seq, make_dummy_pdb(seq)))'''
    
    new_error_handling = '''                    except Exception as e2:
                        logging.error(f"Individual prediction failed for seq len={len(seq)}: {e2}")
                        # 方案1：跳过失败的样本，不生成dummy PDB
                        logging.error(f"跳过序列 {seq[:20]}... (长度: {len(seq)})")
                        continue'''
    
    content = content.replace(old_error_handling, new_error_handling)
    
    # 修改3: 更新OOM处理中的dummy PDB生成
    old_oom_handling = '''                # 单序列仍OOM，生成dummy
                logging.error(f"Single sequence OOM (len={len(sequences[0])}), using dummy PDB")
                return [(seq, make_dummy_pdb(seq)) for seq in sequences]'''
    
    new_oom_handling = '''                # 单序列仍OOM，跳过该序列
                logging.error(f"Single sequence OOM (len={len(sequences[0])}), skipping sequence")
                raise ValueError(f"序列过长导致OOM，跳过序列: {len(sequences[0])}个氨基酸")'''
    
    content = content.replace(old_oom_handling, new_oom_handling)
    
    # 修改4: 更新主函数中的错误处理
    old_main_error = '''        except Exception as e:
            logging.error(f"Batch prediction failed: {e}")
            # 单独处理失败的batch
            for idx, proc_seq, sample_id, pdb_path, meta in processed_batch:
                dummy_content = make_dummy_pdb(proc_seq)
                with open(pdb_path, 'w') as f:
                    f.write(dummy_content)
                
                if args.use_sample_manager:
                    sample_manager.log_failure(sample_id, "structure_prediction", str(e), len(proc_seq))
                
                meta['status'] = 'dummy_fallback'
                meta['error'] = str(e)
                report.append(meta)'''
    
    new_main_error = '''        except Exception as e:
            logging.error(f"Batch prediction failed: {e}")
            # 方案1：跳过失败的batch，记录失败信息
            for idx, proc_seq, sample_id, pdb_path, meta in processed_batch:
                if args.use_sample_manager:
                    sample_manager.log_failure(sample_id, "structure_prediction", str(e), len(proc_seq))
                
                meta['status'] = 'failed'
                meta['error'] = str(e)
                report.append(meta)
                logging.error(f"[{idx+1}/{len(sequences)}] Failed {sample_id}: {str(e)}")'''
    
    content = content.replace(old_main_error, new_main_error)
    
    # 修改5: 更新文档字符串
    old_doc = '''7. Dummy PDB generator for failed predictions'''
    new_doc = '''7. Skip failed samples instead of using dummy PDB to ensure data quality'''
    content = content.replace(old_doc, new_doc)
    
    # 保存修改后的文件
    with open('generate_pdb.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 方案1修改已应用到 generate_pdb.py")
    print("主要更改:")
    print("1. 跳过ESMFold输出为空的序列")
    print("2. 跳过PDB转换失败的序列")
    print("3. 跳过OOM的序列")
    print("4. 记录失败信息但不生成dummy PDB")
    print("5. 确保只有有效的PDB结构进入后续预测")

if __name__ == "__main__":
    apply_solution1()

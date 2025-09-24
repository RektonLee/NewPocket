#!/usr/bin/env python3
"""
测试修复后的预测管道
验证PDB文件存在性检查是否正常工作
"""

import os
import pandas as pd
from pred_fullpipeline import predict_kinetics

def test_prediction_fix():
    """测试修复后的预测功能"""
    
    # 创建测试数据 - 包含有PDB和没有PDB的样本
    test_data = pd.DataFrame({
        'sample_id': ['km_test_0004', 'km_test_0001', 'km_test_0006'],  # 0004和0006有PDB，0001没有
        'sequence': [
            'MSGKLTVITGPMYSGKTTELLSFVEIYKLGKKKVAVFKPKIDSRYHSTMIVSHSGNGVEAHVIERPEEMRKYIEEDTRGVFIDEVQFFNPSLFEVVKDLLDRGIDVFCAGLDLTHKQNPFETTALLLSLADTVIKKKAVCHRCGEYNATLTLKVAGGEEEIDVGGQEKYIAVCRDCYNTLKKRV',
            'MKAILTTGALLCAMTAFAQVPQLNRNNIDEVLKAMTLEEKITLVVGANRYVGDENGPGPAPGMPERKSVDMSGLVEQPKSDGVTAFSSGRVKGAAGDVVPVERLGITTMVLADGPAGLRIDAVRPGDDNTYFCTAFPIGSLLSASWDTGLVERVTAAMGNEVLEYGADVLLAPAMNIHRNPLCGRNFEYYSEDPLLAGKIAAAYVRGVQGNGVGTSVKHFAANSQETLRNGQNASVSERALREIYLKGFEIVVKEAQPWTIMSSYNKINGVLSSENRWLLTDVLRGEWGFKGFVMTDWWAEENGARQIAAGNDMLMPGTPHQYDDILDAVQSGRLDIRFLDDCVRRILQVMVESPTFKRYAYSNKPDLAAHAQVTREAAAQGMVLLKNESALPLVRKSKVALFGVPSYDTMVGGSGSGYVNRAYKVTVDAGLEAAGFRLDKQLAESYRDYVKHEKAKQPAEYFWIIPTVQETLISREVAQAAAKRNEVCVYSIGRMAGEGGDRTLTPGDWYLSETEQANIDLLCETFHKAGKKVIVLLNMGNIVDMGWSDQPDAILHTWMDGQEAGNSVADILAGKVSPSGKLPMTIAKSYEDYSSAKDFPMSNGNPGDVNYDEDIFVGYRHFDRHPETILYPFGFGLSYTDFAYSDLKAEREGDELKISVKVTNTGKRPGREAVQIYVGAPAGNAVKPVKELRAFGKTSELKPGASEVLTMTVKLADLRWFDAYEQAWKLDAGEYVISAAASSRDIRQNVTVTL',
            'MFLDTTPFRADEPYDVFIAGSGPIGATFAKLCVDANLRVCMVEIGAADSFTSKPMKGDPNAPRSVQFGPGQVPIPGYHKKNEIEYQKDIDRFVNVIKGALSTCSIPTSNNHIATLDPSVVSNSLDKPFISLGKNPAQNPFVNLGAEAVTRGVGGMSTHWTCATPEFFAPADFNAPHRERPKLSTDAAEDARIWKDLYAQAKEIIGTSTTEFDHSIRHNLVLRKYNDIFQKENVIREFSPLPLACHRLTDPDYVEWHATDRILEELFTDPVKRGRFTLLTNHRCTKLVFKHYRPGEENEVDYALVEDLLPHMQNPGNPASVKKIYARSYVVACGAVATAQVLANSHIPPDDVVIPFPGGEKGSGGGERDATIPTPLMPMLGKYITEQPMTFCQVVLDSSLMEVVRNPPWPGLDWWKEKVARHVEAFPNDPIPIPFRDPEPQVTIKFTEEHPWHVQIHRDAFSYGAVAENMDTRVIVDYRFFGYTEPQEANELVFQQHYRDAYDMPQPTFKFTMSQDDRARARRMMDDMCNIALKIGGYLPGSEPQFMTPGLALHLAGTTRCGLDTQKTVGNTHCKVHNFNNLYVGGNGVIETGFAANPTLTSICYAIRASNDIIAKFGRHRG'
        ],
        'smiles': [
            'Cc1cn([C@H]2CC[C@@H](CO)O2)c(=O)[nH]c1=O',
            'O=[N+]([O-])c1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1',
            'O=C(CO)[C@@H](O)[C@H](O)[C@H](O)CO'
        ]
    })
    
    # 检查哪些样本有PDB文件
    print("检查PDB文件存在性:")
    for _, row in test_data.iterrows():
        sample_id = row['sample_id']
        pdb_path = f"sample_data/samples/{sample_id}/{sample_id}_protein.pdb"
        exists = os.path.exists(pdb_path)
        print(f"  {sample_id}: {'✅ 存在' if exists else '❌ 不存在'}")
    
    # 模型路径（需要根据实际情况调整）
    model_path = "/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请检查模型路径或使用其他可用的模型")
        return
    
    try:
        print("\n开始测试预测管道...")
        results = predict_kinetics(
            input_data=test_data,
            model_path=model_path,
            output_dir='test_fix_results',
            temperature=303.15,
            use_sample_manager=True,
            sample_data_dir='sample_data'
        )
        
        print("✅ 预测测试完成！")
        print(f"结果数量: {len(results)}")
        
        # 分析结果
        valid_results = results.dropna(subset=['kcat_pred', 'km_pred'])
        failed_results = results[results['kcat_pred'].isna() | results['km_pred'].isna()]
        
        print(f"\n结果分析:")
        print(f"  成功预测: {len(valid_results)} 个样本")
        print(f"  跳过/失败: {len(failed_results)} 个样本")
        
        if len(valid_results) > 0:
            print(f"\n成功预测的样本:")
            for _, result in valid_results.iterrows():
                print(f"  {result['sample_id']}: kcat={result['kcat_pred']:.2f}, Km={result['km_pred']:.2f}")
        
        if len(failed_results) > 0:
            print(f"\n跳过/失败的样本:")
            for _, result in failed_results.iterrows():
                print(f"  {result['sample_id']}: {result.get('error', '未知错误')}")
        
        # 检查输出文件
        print(f"\n输出文件:")
        output_files = [
            'test_fix_results/predictions.csv',
            'test_fix_results/successful_predictions.csv', 
            'test_fix_results/failed_predictions.csv',
            'test_fix_results/prediction_stats.csv'
        ]
        
        for file_path in output_files:
            if os.path.exists(file_path):
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} (未生成)")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction_fix()

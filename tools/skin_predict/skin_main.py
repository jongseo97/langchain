# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:37:39 2024

@author: jspark
"""

from .skin_data import  Dataloader
from .skin_model_utils import load_config, load_model
from .skin_predict import predict


def skin_main(smiles_list):
    script_path = f'./tools/skin_predict'
    model_path = f'{script_path}/model'
    
    
    KE_list = [1,2,3,4]
    
    
    
    # load smiles
    # smiles_list = load_smiles(input_path)
    # smiles_list = check_smiles(smiles_list)
    
    output_dict = {}
    for KE in KE_list:
        # load configuration
        config = load_config(model_path, KE)
        
        # load dataloader
        dataloader = Dataloader(smiles_list, config)
        
        # load model
        model = load_model(model_path, config, KE)
        
        
        # make prediction
        probs, labels, test_vectors_mf, test_vectors_lf = predict(model, dataloader)
        if int(labels) == 1:
            label = 'active'
        else:
            label = 'inactive'
        output_dict[f'KE{KE}'] = label

    return output_dict


# script version  
if __name__ == '__main__':
    
    KE = 'All'  # 1, 2, 3, 4, All
    # KE = 1
    script_path = r'C:\Users\user\Desktop\1\Modeling\18. AOP 예측모델\피부과민성\통합 script'
    input_path = rf'C:\Users\user\Desktop\1\Modeling\18. AOP 예측모델\피부과민성\검증\KE_total.xlsx'
    output_path = rf'C:\Users\user\Desktop\1\Modeling\18. AOP 예측모델\피부과민성\검증\KE{KE}_output.xlsx'
    
    input_path = rf'C:\Users\user\Desktop\1\Modeling\18. AOP 예측모델\피부과민성\통합 script\examples\KE1_test_examples.xlsx'
    output_path = rf'C:\Users\user\Desktop\1\Modeling\18. AOP 예측모델\피부과민성\통합 script\examples\test.xlsx'
    
    df = main(script_path, input_path, output_path, KE)
    
    
# argument version

# import argparse

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='AOP 예측 모델 스크립트')
    
#     # 명령줄 인수 정의
#     parser.add_argument('--model_path', type=str, required=True, help='모델 경로')
#     parser.add_argument('--input_path', type=str, required=True, help='입력 파일 경로')
#     parser.add_argument('--output_path', type=str, required=True, help='출력 파일 경로')
#     parser.add_argument('--KE', type=int, default=2, help='KE 값 (기본값: 2)')
    
#     # 인수 파싱
#     args = parser.parse_args()
    
#     # main 함수 호출
#     df = main(args.model_path, args.input_path, args.output_path, args.KE)

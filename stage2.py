"""
Stage2: prior learning

run `python stage2.py`
"""
import copy
from argparse import ArgumentParser

import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import build_data_pipeline
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from preprocessing.preprocess_ucr import DatasetImporterUCR

from experiments.exp_maskgit import ExpMaskGIT  # MaskGIT 실험 클래스 임포트
from evaluation.evaluation import Evaluation  # 평가 클래스 임포트
from utils import get_root_dir, load_yaml_param_settings, save_model  # 유틸리티 함수들 임포트

def load_args():
    parser = ArgumentParser()  # ArgumentParser 객체 생성
    parser.add_argument('--config', type=str, help="Path to the config data  file.", 
                        default=get_root_dir().joinpath('configs', 'config.yaml'))  # 구성 파일 경로 인자
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')  # 데이터셋 이름 인자
    parser.add_argument('--gpu_device_idx', default=0, type=int)  # GPU 디바이스 인덱스 인자
    return parser.parse_args()  # 인자 파싱 후 반환

def train_stage2(config: dict, 
                 dataset_name: str, 
                 train_data_loader: DataLoader, 
                 test_data_loader: DataLoader, 
                 gpu_device_idx, 
                 do_validate: bool):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = 'TimeVQVAE-stage2'  # 프로젝트 이름

    # fit
    n_classes = len(np.unique(train_data_loader.dataset.Y))  # 클래스 개수
    input_length = train_data_loader.dataset.X.shape[-1]  # 입력 길이
    train_exp = ExpMaskGIT(dataset_name, input_length, config, len(train_data_loader.dataset), n_classes)  # ExpMaskGIT 객체 생성
    config_ = copy.deepcopy(config)  # 구성 파일 깊은 복사
    config_['dataset']['dataset_name'] = dataset_name  # 데이터셋 이름 설정
    wandb_logger = WandbLogger(project=project_name, name=None, config=config_)  # WandbLogger 객체 생성
    trainer = pl.Trainer(logger=wandb_logger, 
                         enable_checkpointing=False, 
                         callbacks=[LearningRateMonitor(logging_interval='epoch')], 
                         max_epochs=config['trainer_params']['max_epochs']['stage2'], 
                         devices=[gpu_device_idx], 
                         accelerator='gpu')  # PyTorch Lightning 트레이너 객체 생성
    trainer.fit(train_exp, 
                train_dataloaders=train_data_loader, 
                val_dataloaders=test_data_loader if do_validate else None)  # 모델 학습

    # 추가 로그 기록
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)  # 학습 가능한 파라미터 수 계산
    wandb.log({'n_trainable_params:': n_trainable_params})  # 파라미터 수 로그 기록

    print('saving the model...')
    save_model({'maskgit': train_exp.maskgit}, id=dataset_name)  # 모델 저장

    # 테스트
    print('evaluating...')
    input_length = train_data_loader.dataset.X.shape[-1]  # 입력 길이
    n_classes = len(np.unique(train_data_loader.dataset.Y))  # 클래스 개수
    evaluation = Evaluation(dataset_name, gpu_device_idx, config)  # 평가 객체 생성
    _, _, x_gen = evaluation.sample(max(evaluation.X_test.shape[0], config['dataset']['batch_sizes']['stage2']), 
                                    input_length, 
                                    n_classes, 
                                    'unconditional')  # 샘플 생성
    z_test, z_gen = evaluation.compute_z(x_gen)  # z 값 계산
    fid, (z_test, z_gen) = evaluation.fid_score(z_test, z_gen)  # FID 점수 계산
    IS_mean, IS_std = evaluation.inception_score(x_gen)  # Inception Score 계산
    wandb.log({'FID': fid, 'IS_mean': IS_mean, 'IS_std': IS_std})  # 결과 로그 기록

    evaluation.log_visual_inspection(min(200, evaluation.X_test.shape[0]), x_gen)  # 시각적 검증 로그 기록
    evaluation.log_pca(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)  # PCA 로그 기록
    evaluation.log_tsne(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)  # t-SNE 로그 기록
    wandb.finish()  # WandB 마무리

if __name__ == '__main__':
    # 구성 파일 로드
    args = load_args()  # 인자 로드
    config = load_yaml_param_settings(args.config)  # 구성 파일 로드

    # 구성 설정
    dataset_names = args.dataset_names  # 데이터셋 이름 목록 설정

    # 실행
    for dataset_name in dataset_names:  # 각 데이터셋 이름에 대해
        # 데이터 파이프라인
        dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])  # 데이터셋 임포터 객체 생성
        batch_size = config['dataset']['batch_sizes']['stage2']  # 배치 크기 설정
        train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]  # 데이터 로더 생성

        # 학습
        train_stage2(config, dataset_name, train_data_loader, test_data_loader, args.gpu_device_idx, do_validate=False)  # Stage 2 학습 실행
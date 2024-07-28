import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from experiments.exp_base import ExpBase, detach_the_unnecessary
from generators.maskgit import MaskGIT


class ExpMaskGIT(ExpBase):
    def __init__(self, dataset_name: str, input_length: int, config: dict, n_train_samples: int, n_classes: int):
        super().__init__()
        self.config = config
        self.maskgit = MaskGIT(dataset_name, input_length, **config['MaskGIT'], config=config, n_classes=n_classes)
        self.T_max = config['trainer_params']['max_epochs']['stage2'] * (np.ceil(n_train_samples / config['dataset']['batch_sizes']['stage2']) + 1)

    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits, target = self.maskgit(x, y)
        logits_l, logits_h = logits
        target_l, target_h = target
        prior_loss_l = F.cross_entropy(logits_l.reshape(-1, logits_l.size(-1)), target_l.reshape(-1))
        prior_loss_h = F.cross_entropy(logits_h.reshape(-1, logits_h.size(-1)), target_h.reshape(-1))
        prior_loss = (prior_loss_l + prior_loss_h)/2
        loss = (prior_loss_l + prior_loss_h)/2

        # 학습률 스케줄러
        sch = self.lr_schedulers()
        sch.step()

        # 로그 기록
        loss_hist = {'loss': loss,
                     'prior_loss': prior_loss,
                     }

        # MaskGIT 샘플링
        r = np.random.rand()
        if batch_idx == 0 and r <= 0.05:
            self.maskgit.eval()

            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # 무조건 샘플링
            s_l, s_h = self.maskgit.iterative_decoding(device=x.device, class_index=class_index)
            x_new_l = self.maskgit.decode_token_ind_to_timeseries(s_l, 'LF').cpu()
            x_new_h = self.maskgit.decode_token_ind_to_timeseries(s_h, 'HF').cpu()
            x_new = x_new_l + x_new_h

            b = 0
            fig, axes = plt.subplots(3, 1, figsize=(4, 2*3))
            axes[0].plot(x_new_l[b, 0, :])
            axes[1].plot(x_new_h[b, 0, :])
            axes[2].plot(x_new[b, 0, :])
            axes[0].set_ylim(-4, 4)
            axes[1].set_ylim(-4, 4)
            axes[2].set_ylim(-4, 4)
            plt.title(f'ep_{self.current_epoch}; class-{class_index}')
            plt.tight_layout()
            wandb.log({f"maskgit sample": wandb.Image(plt)})
            plt.close()

        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits, target = self.maskgit(x, y)
        logits_l, logits_h = logits
        target_l, target_h = target
        prior_loss_l = F.cross_entropy(logits_l.reshape(-1, logits_l.size(-1)), target_l.reshape(-1))
        prior_loss_h = F.cross_entropy(logits_h.reshape(-1, logits_h.size(-1)), target_h.reshape(-1))
        prior_loss = (prior_loss_l + prior_loss_h)/2
        loss = (prior_loss_l + prior_loss_h)/2

        # 로그 기록
        loss_hist = {'loss': loss,
                     'prior_loss': prior_loss,
                     }

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.maskgit.parameters(), 'lr': self.config['exp_params']['LR']},
                                 ],
                                weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits, target = self.maskgit(x, y)
        logits_l, logits_h = logits
        target_l, target_h = target
        prior_loss_l = F.cross_entropy(logits_l.reshape(-1, logits_l.size(-1)), target_l.reshape(-1))
        prior_loss_h = F.cross_entropy(logits_h.reshape(-1, logits_h.size(-1)), target_h.reshape(-1))
        prior_loss = (prior_loss_l + prior_loss_h)/2
        loss = (prior_loss_l + prior_loss_h)/2

        # 로그 기록
        loss_hist = {'loss': loss,
                     'prior_loss': prior_loss,
                     }

        detach_the_unnecessary(loss_hist)
        return loss_hist

이 코드도 한국말로 자세한 설명과 주석 부탁해.
이 코드는 MaskGIT 모델을 사용하여 훈련, 검증 및 테스트를 수행하는 실험 클래스 ExpMaskGIT을 정의하고 있습니다. 주석과 함께 각 부분을 자세히 설명하겠습니다.

python
Copy code
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from experiments.exp_base import ExpBase, detach_the_unnecessary
from generators.maskgit import MaskGIT

class ExpMaskGIT(ExpBase):
    def __init__(self, dataset_name: str, input_length: int, config: dict, n_train_samples: int, n_classes: int):
        """
        ExpMaskGIT 클래스의 초기화 메소드.
        
        :param dataset_name: 데이터셋 이름
        :param input_length: 입력 길이
        :param config: 설정 값이 포함된 딕셔너리
        :param n_train_samples: 훈련 샘플 수
        :param n_classes: 클래스 수
        """
        super().__init__()
        self.config = config
        self.maskgit = MaskGIT(dataset_name, input_length, **config['MaskGIT'], config=config, n_classes=n_classes)
        self.T_max = config['trainer_params']['max_epochs']['stage2'] * (np.ceil(n_train_samples / config['dataset']['batch_sizes']['stage2']) + 1)

    def forward(self, x):
        """
        Forward 메소드. 현재 구현되지 않았습니다.
        
        :param x: (B, C, L) 입력 텐서
        """
        pass

    def training_step(self, batch, batch_idx):
        """
        훈련 단계에서 호출되는 메소드.
        
        :param batch: 배치 데이터
        :param batch_idx: 배치 인덱스
        :return: 손실 히스토리 딕셔너리
        """
        x, y = batch

        logits, target = self.maskgit(x, y)
        logits_l, logits_h = logits
        target_l, target_h = target
        prior_loss_l = F.cross_entropy(logits_l.reshape(-1, logits_l.size(-1)), target_l.reshape(-1))
        prior_loss_h = F.cross_entropy(logits_h.reshape(-1, logits_h.size(-1)), target_h.reshape(-1))
        prior_loss = (prior_loss_l + prior_loss_h) / 2
        loss = (prior_loss_l + prior_loss_h) / 2

        # 학습률 스케줄러 업데이트
        sch = self.lr_schedulers()
        sch.step()

        # 손실 기록
        loss_hist = {'loss': loss,
                     'prior_loss': prior_loss}

        # MaskGIT 샘플링
        r = np.random.rand()
        if batch_idx == 0 and r <= 0.05:
            self.maskgit.eval()

            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # 무조건 샘플링
            s_l, s_h = self.maskgit.iterative_decoding(device=x.device, class_index=class_index)
            x_new_l = self.maskgit.decode_token_ind_to_timeseries(s_l, 'LF').cpu()
            x_new_h = self.maskgit.decode_token_ind_to_timeseries(s_h, 'HF').cpu()
            x_new = x_new_l + x_new_h

            b = 0
            fig, axes = plt.subplots(3, 1, figsize=(4, 2 * 3))
            axes[0].plot(x_new_l[b, 0, :])
            axes[1].plot(x_new_h[b, 0, :])
            axes[2].plot(x_new[b, 0, :])
            axes[0].set_ylim(-4, 4)
            axes[1].set_ylim(-4, 4)
            axes[2].set_ylim(-4, 4)
            plt.title(f'ep_{self.current_epoch}; class-{class_index}')
            plt.tight_layout()
            wandb.log({f"maskgit sample": wandb.Image(plt)})
            plt.close()

        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        검증 단계에서 호출되는 메소드.
        
        :param batch: 배치 데이터
        :param batch_idx: 배치 인덱스
        :return: 손실 히스토리 딕셔너리
        """
        x, y = batch

        logits, target = self.maskgit(x, y)
        logits_l, logits_h = logits
        target_l, target_h = target
        prior_loss_l = F.cross_entropy(logits_l.reshape(-1, logits_l.size(-1)), target_l.reshape(-1))
        prior_loss_h = F.cross_entropy(logits_h.reshape(-1, logits_h.size(-1)), target_h.reshape(-1))
        prior_loss = (prior_loss_l + prior_loss_h) / 2
        loss = (prior_loss_l + prior_loss_h) / 2

        # 손실 기록
        loss_hist = {'loss': loss,
                     'prior_loss': prior_loss}

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        """
        최적화 도구 설정 메소드.
        
        :return: 최적화 도구와 학습률 스케줄러 딕셔너리
        """
        opt = torch.optim.AdamW([{'params': self.maskgit.parameters(), 'lr': self.config['exp_params']['LR']}],
                                weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        """
        테스트 단계에서 호출되는 메소드.
        
        :param batch: 배치 데이터
        :param batch_idx: 배치 인덱스
        :return: 손실 히스토리 딕셔너리
        """
        x, y = batch

        logits, target = self.maskgit(x, y)
        logits_l, logits_h = logits
        target_l, target_h = target
        prior_loss_l = F.cross_entropy(logits_l.reshape(-1, logits_l.size(-1)), target_l.reshape(-1))
        prior_loss_h = F.cross_entropy(logits_h.reshape(-1, logits_h.size(-1)), target_h.reshape(-1))
        prior_loss = (prior_loss_l + prior_loss_h) / 2
        loss = (prior_loss_l + prior_loss_h) / 2

        # 손실 기록
        loss_hist = {'loss': loss,
                     'prior_loss': prior_loss}

        detach_the_unnecessary(loss_hist)
        return loss_hist
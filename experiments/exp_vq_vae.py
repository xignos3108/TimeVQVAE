import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import wandb
import pytorch_lightning as pl

# VQVAE의 Encoder와 Decoder 모듈을 임포트
from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from experiments.exp_base import ExpBase, detach_the_unnecessary
from vector_quantization import VectorQuantize
from supervised_FCN_2.example_pretrained_model_loading import load_pretrained_FCN
from utils import compute_downsample_rate, freeze, timefreq_to_time, time_to_timefreq, zero_pad_low_freq, zero_pad_high_freq, quantize


class ExpVQVAE(pl.LightningModule):
    def __init__(self,
                 input_length: int,  # 입력 시퀀스의 길이
                 config: dict):  # 구성 파일
        """
        :param input_length: 길이의 입력 시퀀스
        :param config: 구성 파일(config.yaml)
        :param n_train_samples: 학습 샘플 수
        """
        super().__init__()
        self.config = config  # 구성 파일 저장

        self.n_fft = config['VQ-VAE']['n_fft']  # VQ-VAE 구성에서 n_fft 설정
        dim = config['encoder']['dim']  # 인코더 차원
        in_channels = config['dataset']['in_channels']  # 입력 채널 수
        downsampled_width_l = config['encoder']['downsampled_width']['lf']  # 저주파수 대역 다운샘플링 너비
        downsampled_width_h = config['encoder']['downsampled_width']['hf']  # 고주파수 대역 다운샘플링 너비
        downsample_rate_l = compute_downsample_rate(input_length, self.n_fft, downsampled_width_l)  # 저주파수 대역 다운샘플링 비율 계산
        downsample_rate_h = compute_downsample_rate(input_length, self.n_fft, downsampled_width_h)  # 고주파수 대역 다운샘플링 비율 계산

        # 인코더 생성
        self.encoder_l = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_l, config['encoder']['n_resnet_blocks'], frequency_indepence=False)
        self.encoder_h = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_h, config['encoder']['n_resnet_blocks'], frequency_indepence=True)

        # 벡터 양자화 모듈 생성
        self.vq_model_l = VectorQuantize(dim, config['VQ-VAE']['codebook_sizes']['lf'], **config['VQ-VAE'])
        self.vq_model_h = VectorQuantize(dim, config['VQ-VAE']['codebook_sizes']['hf'], **config['VQ-VAE'])

        # 디코더 생성
        self.decoder_l = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_l, config['decoder']['n_resnet_blocks'], frequency_indepence=False)
        self.decoder_h = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_h, config['decoder']['n_resnet_blocks'], frequency_indepence=True)

    def forward(self, batch, batch_idx, return_x_rec:bool=False):
        """
        :param x: 입력 시퀀스 (B, C, L)
        """
        x, y = batch  # 배치에서 입력 데이터와 타겟 데이터 분리

        recons_loss = {'LF.time': 0., 'HF.time': 0., 'LF.timefreq': 0., 'HF.timefreq': 0., 'perceptual': 0.}
        vq_losses = {'LF': None, 'HF': None}
        perplexities = {'LF': 0., 'HF': 0.}

        # 시간-주파수 변환: STFT(x)
        C = x.shape[1]  # 입력 데이터의 채널 수
        xf = time_to_timefreq(x, self.n_fft, C)  # STFT 변환 (B, C, H, W)
        u_l = zero_pad_high_freq(xf)  # 고주파수 대역 제로 패딩 (B, C, H, W)
        x_l = timefreq_to_time(u_l, self.n_fft, C)  # 시간 영역으로 변환 (B, C, L)

        # 디코더에 'upsample_size' 등록
        for decoder in [self.decoder_l, self.decoder_h]:
            if not decoder.is_upsample_size_updated:
                decoder.register_upsample_size(torch.IntTensor(np.array(xf.shape[2:])))

        # 저주파수 대역 인코딩
        z_l = self.encoder_l(u_l)  # 인코딩
        z_q_l, indices_l, vq_loss_l, perplexity_l = quantize(z_l, self.vq_model_l)  # 양자화
        xfhat_l = self.decoder_l(z_q_l)  # 디코딩
        uhat_l = zero_pad_high_freq(xfhat_l)  # 고주파수 대역 제로 패딩
        xhat_l = timefreq_to_time(uhat_l, self.n_fft, C)  # 시간 영역으로 변환 (B, C, L)

        # 고주파수 대역 인코딩
        u_h = zero_pad_low_freq(xf)  # 저주파수 대역 제로 패딩 (B, C, H, W)
        x_h = timefreq_to_time(u_h, self.n_fft, C)  # 시간 영역으로 변환 (B, C, L)

        z_h = self.encoder_h(u_h)  # 인코딩
        z_q_h, indices_h, vq_loss_h, perplexity_h = quantize(z_h, self.vq_model_h)  # 양자화
        xfhat_h = self.decoder_h(z_q_h)  # 디코딩
        uhat_h = zero_pad_low_freq(xfhat_h)  # 저주파수 대역 제로 패딩
        xhat_h = timefreq_to_time(uhat_h, self.n_fft, C)  # 시간 영역으로 변환 (B, C, L)

        if return_x_rec:
            x_rec = xhat_l + xhat_h  # 재구성된 입력 데이터 (b c l)
            return x_rec  # (b c l)

        recons_loss['LF.time'] = F.mse_loss(x_l, xhat_l)  # 저주파수 대역 시간 영역 손실
        recons_loss['LF.timefreq'] = F.mse_loss(u_l, uhat_l)  # 저주파수 대역 시간-주파수 영역 손실
        perplexities['LF'] = perplexity_l  # 저주파수 대역 perplexity
        vq_losses['LF'] = vq_loss_l  # 저주파수 대역 벡터 양자화 손실

        recons_loss['HF.time'] = F.l1_loss(x_h, xhat_h)  # 고주파수 대역 시간 영역 손실
        recons_loss['HF.timefreq'] = F.mse_loss(u_h, uhat_h)  # 고주파수 대역 시간-주파수 영역 손실
        perplexities['HF'] = perplexity_h  # 고주파수 대역 perplexity
        vq_losses['HF'] = vq_loss_h  # 고주파수 대역 벡터 양자화 손실

        # 입력 데이터와 재구성된 데이터 플롯
        if not self.training and batch_idx == 0:
            b = np.random.randint(0, x_h.shape[0])
            c = np.random.randint(0, x_h.shape[1])

            alpha = 0.7
            fig, axes = plt.subplots(3, 1, figsize=(4, 2*3))
            plt.suptitle(f'step-{self.global_step} \n (blue:GT, orange:reconstructed)')
            axes[0].plot(x_l[b, c].cpu(), alpha=alpha)
            axes[0].plot(xhat_l[b, c].detach().cpu(), alpha=alpha)
            axes[0].set_title(r'$x_l$ (LF)')
            axes[0].set_ylim(-4, 4)

            axes[1].plot(x_h[b, c].cpu(), alpha=alpha)
            axes[1].plot(xhat_h[b, c].detach().cpu(), alpha=alpha)
            axes[1].set_title(r'$x_h$ (HF)')
            axes[1].set_ylim(-4, 4)

            axes[2].plot(x_l[b, c].cpu() + x_h[b, c].cpu(), alpha=alpha)
            axes[2].plot(xhat_l[b, c].detach().cpu() + xhat_h[b, c].detach().cpu(), alpha=alpha)
            axes[2].set_title(r'$x$ (LF+HF)')
            axes[2].set_ylim(-4, 4)

            plt.tight_layout()
            wandb.log({"x vs x_rec (val)": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_losses, perplexities

    def training_step(self, batch, batch_idx):
        recons_loss, vq_losses, perplexities = self.forward(batch, batch_idx)
        loss = (recons_loss['LF.time'] + recons_loss['HF.time'] +
                recons_loss['LF.timefreq'] + recons_loss['HF.timefreq']) + \
                vq_losses['LF']['loss'] + vq_losses['HF']['loss'] + \
                recons_loss['perceptual']

        # 학습률 스케줄러
        sch = self.lr_schedulers()
        sch.step()

        # 로그 기록
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
                     'recons_loss.LF.time': recons_loss['LF.time'],
                     'recons_loss.HF.time': recons_loss['HF.time'],

                     'recons_loss.LF.timefreq': recons_loss['LF.timefreq'],
                     'recons_loss.HF.timefreq': recons_loss['HF.timefreq'],

                     'commit_loss.LF': vq_losses['LF']['commit_loss'],
                     'commit_loss.HF': vq_losses['HF']['commit_loss'],
                     'perplexity.LF': perplexities['LF'],
                     'perplexity.HF': perplexities['HF'],

                     'perceptual': recons_loss['perceptual']
                     }
        
        # 로그 기록
        for k in loss_hist.keys():
            self.log(f'train/{k}', loss_hist[k])

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        recons_loss, vq_losses, perplexities = self.forward(batch, batch_idx)
        loss = (recons_loss['LF.time'] + recons_loss['HF.time'] +
                recons_loss['LF.timefreq'] + recons_loss['HF.timefreq']) + \
                vq_losses['LF']['loss'] + vq_losses['HF']['loss'] + \
                recons_loss['perceptual']

        # 로그 기록
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
                     'recons_loss.LF.time': recons_loss['LF.time'],
                     'recons_loss.HF.time': recons_loss['HF.time'],

                     'recons_loss.LF.timefreq': recons_loss['LF.timefreq'],
                     'recons_loss.HF.timefreq': recons_loss['HF.timefreq'],

                     'commit_loss.LF': vq_losses['LF']['commit_loss'],
                     'commit_loss.HF': vq_losses['HF']['commit_loss'],
                     'perplexity.LF': perplexities['LF'],
                     'perplexity.HF': perplexities['HF'],

                     'perceptual': recons_loss['perceptual']
                     }
        
        # 로그 기록
        for k in loss_hist.keys():
            self.log(f'val/{k}', loss_hist[k])

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['LR'])  # AdamW 옵티마이저 설정
        T_max = self.config['trainer_params']['max_steps']['stage1']  # 스케줄러의 최대 단계 설정
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, T_max, eta_min=1e-5)}  # 옵티마이저와 학습률 스케줄러 반환
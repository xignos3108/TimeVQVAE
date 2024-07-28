import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),  # 비선형 활성화 함수
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),  # 첫 번째 컨볼루션 레이어
            nn.LeakyReLU(),  # 비선형 활성화 함수
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)  # 두 번째 컨볼루션 레이어
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))  # 배치 정규화 레이어 추가
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)  # 입력과 변환된 출력을 더하여 반환


class VQVAEEncBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1),
                      padding_mode='replicate'),  # 컨볼루션 레이어
            nn.BatchNorm2d(out_channels),  # 배치 정규화 레이어
            nn.LeakyReLU(inplace=True))  # 비선형 활성화 함수

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1)),  # 컨볼루션 트랜스포즈 레이어
            nn.BatchNorm2d(out_channels),  # 배치 정규화 레이어
            nn.LeakyReLU(inplace=True))  # 비선형 활성화 함수

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEEncoder(nn.Module):
    """
    VQ-VAE 논문에서 구현된 인코더를 따릅니다.
    """

    def __init__(self, d: int, num_channels: int, downsample_rate: int, n_resnet_blocks: int, bn: bool = True, **kwargs):
        """
        :param d: 숨겨진 차원 크기
        :param num_channels: 입력 채널 크기
        :param downsample_rate: 2의 배수여야 함; 예: 2, 4, 8, 16, ...
        :param n_resnet_blocks: ResNet 블록 수
        :param bn: 배치 정규화 사용 여부
        :param kwargs: 기타 인자
        """
        super().__init__()
        self.encoder = nn.Sequential(
            VQVAEEncBlock(num_channels, d),
            *[VQVAEEncBlock(d, d) for _ in range(int(np.log2(downsample_rate)) - 1)],
            *[nn.Sequential(ResBlock(d, d, bn=bn), nn.BatchNorm2d(d)) for _ in range(n_resnet_blocks)],
        )

        self.is_num_tokens_updated = False
        self.register_buffer('num_tokens', torch.zeros(1).int())
        self.register_buffer('H_prime', torch.zeros(1).int())
        self.register_buffer('W_prime', torch.zeros(1).int())

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return (B, C, H, W') where W' <= W
        """
        out = self.encoder(x)
        if not self.is_num_tokens_updated:
            self.H_prime += out.shape[2]
            self.W_prime += out.shape[3]
            self.num_tokens += self.H_prime * self.W_prime
            self.is_num_tokens_updated = True
        return out


class VQVAEDecoder(nn.Module):
    """
    VQ-VAE 논문에서 구현된 디코더를 따릅니다.
    """

    def __init__(self, d: int, num_channels: int, downsample_rate: int, n_resnet_blocks: int, **kwargs):
        """
        :param d: 숨겨진 차원 크기
        :param num_channels: 입력 채널 크기
        :param downsample_rate: 2의 배수여야 함; 예: 2, 4, 8, 16, ...
        :param n_resnet_blocks: ResNet 블록 수
        :param kwargs: 기타 인자
        """
        super().__init__()
        self.decoder = nn.Sequential(
            *[nn.Sequential(ResBlock(d, d), nn.BatchNorm2d(d)) for _ in range(n_resnet_blocks)],
            *[VQVAEDecBlock(d, d) for _ in range(int(np.log2(downsample_rate)) - 1)],
            nn.ConvTranspose2d(d, num_channels, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1)),
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1)),  # 추가 업샘플링 레이어
        )

        self.is_upsample_size_updated = False
        self.register_buffer("upsample_size", torch.zeros(2))

    def register_upsample_size(self, hw: torch.IntTensor):
        """
        :param hw: (높이 H, 너비 W) 입력 크기
        """
        self.upsample_size = hw
        self.is_upsample_size_updated = True

    def forward(self, x):
        """
        :param x: 인코더의 출력 (B, C, H, W')
        :return  (B, C, H, W)
        """
        out = self.decoder(x)
        if isinstance(self.upsample_size, torch.Tensor):
            upsample_size = self.upsample_size.cpu().numpy().astype(int)
            upsample_size = [*upsample_size]
            out = F.interpolate(out, size=upsample_size, mode='bilinear', align_corners=True)
            return out
        else:
            raise ValueError('self.upsample_size is not yet registered.')


if __name__ == '__main__':
    import numpy as np

    x = torch.rand(1, 2, 4, 128)  # (배치, 채널, 높이, 너비)

    encoder = VQVAEEncoder(d=32, num_channels=2, downsample_rate=4, n_resnet_blocks=2)
    decoder = VQVAEDecoder(d=32, num_channels=2, downsample_rate=4, n_resnet_blocks=2)
    decoder.upsample_size = torch.IntTensor(np.array(x.shape[2:]))

    z = encoder(x)
    x_recons = decoder(z)

    print('x.shape:', x.shape)
    print('z.shape:', z.shape)
    print('x_recons.shape:', x_recons.shape)
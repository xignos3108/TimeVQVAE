# 필요한 라이브러리들을 임포트
import copy  # 객체의 깊은 복사를 위한 라이브러리
import torch  # PyTorch의 핵심 라이브러리
from torch import nn, einsum  # PyTorch의 신경망 모듈과 아인슈타인 합성 함수
import torch.nn.functional as F  # PyTorch의 함수형 API
import torch.distributed as distributed  # 분산 학습을 위한 PyTorch 모듈
from torch.cuda.amp import autocast  # 자동 혼합 정밀도를 위한 PyTorch 모듈
from torch.distributions.categorical import Categorical  # 범주형 분포를 위한 PyTorch 모듈

from einops import rearrange, repeat  # 텐서 변환을 위한 라이브러리
from contextlib import contextmanager  # 컨텍스트 매니저를 위한 라이브러리

from utils import *  # 유틸리티 함수 임포트
from x_transformers import ContinuousTransformerWrapper, Encoder as TFEncoder, Decoder as TFDecoder  # x_transformers 모듈에서 필요한 클래스 임포트


# 값이 존재하는지 확인하는 함수
def exists(val):
    return val is not None  # 값이 None이 아닌 경우 True를 반환


# 값이 존재하지 않을 경우 기본 값을 반환하는 함수
def default(val, d):
    return val if exists(val) else d  # 값이 None이 아니면 해당 값을, 그렇지 않으면 기본 값 d를 반환


# 아무 동작도 하지 않는 함수
def noop(*args, **kwargs):
    pass  # 아무 동작도 하지 않음


# 텐서를 L2 정규화하는 함수
def l2norm(t):
    return F.normalize(t, p=2, dim=-1)  # 텐서를 L2 노름으로 정규화


# 로그 함수를 안정적으로 계산하기 위한 함수
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))  # 입력 텐서 t에 작은 값을 더해 로그를 계산


# Gumbel 노이즈를 생성하는 함수
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)  # 입력 텐서 t와 같은 형태로 0에서 1 사이의 균일 분포 노이즈 생성
    return -log(-log(noise))  # Gumbel 노이즈를 생성


# Gumbel 샘플링 함수
def gumbel_sample(t, temperature=1., dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)  # 온도가 0이면 최대값의 인덱스를 반환
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)  # Gumbel 노이즈를 추가한 후 최대값의 인덱스를 반환


# 소프트맥스 샘플링 함수
def softmax_sample(t, temperature=1., dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)  # 온도가 0이면 최대값의 인덱스를 반환
    m = Categorical(logits=t / temperature)  # 소프트맥스 분포 생성
    return m.sample()  # 샘플링하여 인덱스를 반환


# 지수 이동 평균을 계산하는 함수
def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))  # 이동 평균 업데이트


# 라플라스 스무딩 함수
def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)  # 라플라스 스무딩 적용


# 샘플에서 벡터를 추출하는 함수
def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device  # 샘플의 수와 장치를 가져옴
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]  # 샘플 수가 충분하면 무작위로 선택
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)  # 그렇지 않으면 중복을 허용하여 무작위로 선택
    return samples[indices]  # 선택된 샘플 반환


# K-평균 클러스터링 함수
def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device  # 샘플의 차원, 데이터 유형, 장치를 가져옴
    means = sample_vectors(samples, num_clusters)  # 초기 중심점 샘플링
    for _ in range(num_iters):  # K-평균 반복
        if use_cosine_sim:
            dists = samples @ means.t()  # 코사인 유사도를 사용하는 경우
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')  # 유클리드 거리를 사용하는 경우
            dists = -(diffs ** 2).sum(dim=-1)  # 거리를 계산
        buckets = dists.max(dim=-1).indices  # 각 샘플이 속한 클러스터를 결정
        bins = torch.bincount(buckets, minlength=num_clusters)  # 각 클러스터의 크기를 계산
        zero_mask = bins == 0  # 빈 클러스터를 찾음
        bins_min_clamped = bins.masked_fill(zero_mask, 1)  # 빈 클러스터를 1로 채움
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)  # 새로운 중심점을 초기화
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)  # 새로운 중심점을 계산
        new_means = new_means / bins_min_clamped[..., None]  # 클러스터 크기로 나누어 평균을 구함
        if use_cosine_sim:
            new_means = l2norm(new_means)  # 코사인 유사도를 사용하는 경우 정규화
        means = torch.where(zero_mask[..., None], means, new_means)  # 빈 클러스터는 업데이트하지 않음
    return means, bins  # 중심점과 클러스터 크기를 반환


# 정규화 손실 함수
def orthgonal_loss_fn(t):
    n = t.shape[0]  # 텐서의 첫 번째 차원 크기
    normed_codes = l2norm(t)  # 텐서를 L2 정규화
    identity = torch.eye(n, device=t.device)  # 단위 행렬 생성
    cosine_sim = einsum('i d, j d -> i j', normed_codes, normed_codes)  # 코사인 유사도 계산
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)  # 손실 계산


# 유클리드 코드북 클래스
class EuclideanCodebook(nn.Module):
    """
    source: https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
    """
    def __init__(
            self,
            dim,  # 코드북 벡터의 차원
            codebook_size,  # 코드북의 크기
            kmeans_init=False,  # K-평균으로 초기화할지 여부
            kmeans_iters=10,  # K-평균 반복 횟수
            decay=0.8,  # 지수 이동 평균의 감쇠율
            eps=1e-5,  # 작은 값, 숫자 안정성을 위해 사용
            threshold_ema_dead_code=2,  # EMA에서 죽은 코드의 임계값
            use_ddp=False,  # 분산 데이터 병렬 처리를 사용할지 여부
            learnable_codebook=False,  # 코드북이 학습 가능한지 여부
            sample_codebook_temp=0,  # 코드북 샘플링 온도
            emb_dropout=0.,  # 드롭아웃 확률
    ):
        super().__init__()  # 부모 클래스 초기화
        self.decay = decay  # 감쇠율 설정
        init_fn = torch.randn if not kmeans_init else torch.zeros  # 초기화 함수 설정
        embed = init_fn(codebook_size, dim)  # 코드북 초기화

        self.codebook_size = codebook_size  # 코드북 크기 설정
        self.kmeans_iters = kmeans_iters  # K-평균 반복 횟수 설정
        self.eps = eps  # 작은 값 설정
        self.threshold_ema_dead_code = threshold_ema_dead_code  # EMA에서 죽은 코드의 임계값 설정
        self.sample_codebook_temp = sample_codebook_temp  # 코드북 샘플링 온도 설정
        self.emb_dropout = emb_dropout  # 드롭아웃 확률 설정

        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop  # 분산 데이터 병렬 처리를 위한 함수 설정
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))  # K-평균 초기화 여부를 버퍼에 저장
        self.register_buffer('cluster_size', torch.zeros(codebook_size))  # 클러스터 크기 버퍼 초기화
        self.register_buffer('embed', embed)  # 코드북을 버퍼에 저장
        self.register_buffer('embed_avg', embed.clone())  # 코드북의 평균값 버퍼 초기화

        self._needs_init = kmeans_init  # K-평균 초기화가 필요한지 여부 설정
        self.learnable_codebook = learnable_codebook  # 코드북이 학습 가능한지 여부 설정
        if self.learnable_codebook:
            self.embed = nn.Parameter(self.embed)  # 코드북을 학습 가능한 매개변수로 설정

    @property  # 프로퍼티로 정의
    def needs_init(self):
        return self._needs_init  # K-평균 초기화가 필요한지 여부 반환

    @torch.no_grad()  # 역전파 중 계산을 하지 않도록 설정
    def init_embed_(self, data):
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)  # K-평균으로 코드북 초기화
        self.embed.data.copy_(embed)  # 코드북 데이터를 복사
        self.embed_avg.data.copy_(embed)  # 코드북 평균 데이터를 복사
        self.cluster_size.data.copy_(cluster_size)  # 클러스터 크기 데이터를 복사

    @torch.no_grad()  # 역전파 중 계산을 하지 않도록 설정
    def init_embed_inplace_(self, data):
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)  # K-평균으로 코드북 초기화
        self.embed.copy_(embed)  # 코드북 데이터를 복사
        self.embed_avg.copy_(embed)  # 코드북 평균 데이터를 복사
        self.cluster_size.copy_(cluster_size)  # 클러스터 크기 데이터를 복사

    def replace(self, samples, mask):
        new_data = sample_vectors(samples, mask.sum().item())  # 샘플 벡터 추출
        self.embed.data[mask] = new_data  # 선택된 코드북 데이터 교체

    def forward(self, x):
        device, dtype = x.device, x.dtype  # 입력 텐서의 장치와 데이터 유형 가져옴
        flatten = rearrange(x, '... d -> (...) d')  # 텐서를 평탄화

        codebook = self.embed if not self.learnable_codebook else self.embed.data  # 코드북 데이터 가져옴
        self.all_reduce_fn(codebook)  # 분산 데이터 병렬 처리

        if self.training and self.emb_dropout > 0:
            keep_indices = torch.empty(self.codebook_size, device=device).uniform_(0, 1) > self.emb_dropout  # 드롭아웃 적용
            codebook = codebook[keep_indices]  # 드롭아웃된 코드북 데이터 가져옴

        dist = (flatten ** 2).sum(dim=1, keepdim=True) - 2 * flatten @ codebook.t() + (codebook.t() ** 2).sum(dim=0, keepdim=True)  # 거리 계산
        embed_ind = dist.argmin(dim=-1)  # 최솟값의 인덱스 찾기
        quantize = F.embedding(embed_ind, codebook)  # 코드북에서 임베딩 찾기

        if self.training:
            embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)  # 원-핫 인코딩
            cluster_size = embed_onehot.sum(dim=0)  # 클러스터 크기 계산

            self.all_reduce_fn(cluster_size)  # 분산 데이터 병렬 처리
            ema_inplace(self.cluster_size, cluster_size, self.decay)  # EMA 업데이트

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps)  # 라플라스 스무딩 적용
            embed_sum = flatten.t() @ embed_onehot  # 임베딩 합계 계산

            self.all_reduce_fn(embed_sum)  # 분산 데이터 병렬 처리
            ema_inplace(self.embed_avg, embed_sum, self.decay)  # EMA 업데이트

            embed_normalized = self.embed_avg / cluster_size  # 임베딩 정규화
            if self.threshold_ema_dead_code != 0:
                dead_mask = self.cluster_size < self.threshold_ema_dead_code  # 죽은 코드 찾기
                self.replace(flatten, dead_mask)  # 죽은 코드 교체

        perplexity = None  # 퍼플렉시티 초기화
        if self.training and self.sample_codebook_temp > 0:
            perplexity = Categorical(logits=dist).entropy()  # 퍼플렉시티 계산

        quantize = rearrange(quantize, '... d -> ... d')  # 텐서 변환
        return quantize, embed_ind, quantize.detach() - flatten.detach(), perplexity  # 양자화된 값, 인덱스, 손실, 퍼플렉시티 반환


# 벡터 양자화 클래스
class VectorQuantize(nn.Module):
    """
    source: https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
    """
    def __init__(
            self,
            dim,  # 입력 차원
            codebook_size,  # 코드북 크기
            codebook_dim=None,  # 코드북 차원 (기본값: 입력 차원)
            heads=1,  # 헤드 수 (멀티헤드 어텐션을 위해)
            kmeans_init=False,  # K-평균 초기화 여부
            kmeans_iters=10,  # K-평균 반복 횟수
            decay=0.8,  # 지수 이동 평균의 감쇠율
            eps=1e-5,  # 작은 값, 숫자 안정성을 위해 사용
            threshold_ema_dead_code=2,  # EMA에서 죽은 코드의 임계값
            channel_last=True,  # 채널이 마지막인지 여부
            accept_image_fmap=False,  # 이미지 피처 맵을 수락할지 여부
            commitment_weight=1.,  # 커밋먼트 손실 가중치
            orthogonal_reg_weight=0.,  # 직교 정규화 손실 가중치
            orthogonal_reg_active_codes_only=False,  # 활성 코드에만 직교 정규화 적용 여부
            orthogonal_reg_max_codes=None,  # 최대 코드 수
            sample_codebook_temp=0.,  # 코드북 샘플링 온도
            sync_codebook=False,  # 코드북 동기화 여부
            emb_dropout=0.,  # 드롭아웃 확률
            **kwargs  # 추가 인수
    ):
        super().__init__()  # 부모 클래스 초기화
        self.heads = heads  # 헤드 수 설정
        codebook_dim = default(codebook_dim, dim)  # 코드북 차원 설정 (기본값: 입력 차원)
        codebook_input_dim = codebook_dim * heads  # 코드북 입력 차원 설정

        requires_projection = codebook_input_dim != dim  # 입력 차원과 코드북 입력 차원이 다른지 확인
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()  # 입력 프로젝션 설정
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()  # 출력 프로젝션 설정

        self.eps = eps  # 작은 값 설정
        self.commitment_weight = commitment_weight  # 커밋먼트 손실 가중치 설정

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0  # 직교 정규화 손실 여부 확인
        self.orthogonal_reg_weight = orthogonal_reg_weight  # 직교 정규화 손실 가중치 설정
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only  # 활성 코드에만 직교 정규화 적용 여부 설정
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes  # 최대 코드 수 설정

        codebook_class = EuclideanCodebook  # 코드북 클래스 설정

        self._codebook = codebook_class(  # EuclideanCodebook 클래스의 인스턴스를 초기화
            dim=codebook_dim,  # 코드북 벡터의 차원
            codebook_size=codebook_size,  # 코드북의 크기 (벡터의 개수)
            kmeans_init=kmeans_init,  # k-means로 코드북 초기화 여부
            kmeans_iters=kmeans_iters,  # k-means 반복 횟수
            decay=decay,  # EMA(지수 이동 평균) 감쇠율
            eps=eps,  # 작은 값 추가하여 수치적으로 안정화
            threshold_ema_dead_code=threshold_ema_dead_code,  # EMA를 통해 갱신되지 않는 코드 임계값
            use_ddp=sync_codebook,  # 분산 학습 시 코드북 동기화 여부
            learnable_codebook=has_codebook_orthogonal_loss,  # 코드북을 학습 가능하게 할지 여부
            sample_codebook_temp=sample_codebook_temp,  # 코드북 샘플링 온도
            emb_dropout=emb_dropout,  # 드롭아웃 확률
        )

        self.codebook_size = codebook_size  # 코드북 크기를 저장

        self.accept_image_fmap = accept_image_fmap  # 입력이 이미지 피처 맵인지 여부
        self.channel_last = channel_last  # 입력의 채널이 마지막인지 여부

    @property  # 속성으로 정의된 메소드
    def codebook(self):  # 코드북 벡터를 반환
        return self._codebook.embed

    def forward(self, x):  # 순전파 메소드
        """
        x: (B, N, D)  # 입력 텐서의 형태 (배치 크기, 시퀀스 길이, 차원)
        """
        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size  # 입력의 shape, device, heads 정보를 저장
        need_transpose = not self.channel_last and not self.accept_image_fmap  # 채널 마지막이 아니고 이미지 피처 맵도 아닌 경우 전치 필요
        vq_loss = {'loss': torch.tensor([0.], device=device, requires_grad=self.training),  # 손실 초기화
                   'commit_loss': 0.,  # 커밋 손실 초기화
                   'orthogonal_reg_loss': 0.,  # 직교 손실 초기화
                   }

        if self.accept_image_fmap:  # 입력이 이미지 피처 맵인 경우
            height, width = x.shape[-2:]  # 높이와 너비 추출
            x = rearrange(x, 'b c h w -> b (h w) c')  # (B, C, H, W) -> (B, H*W, C)

        if need_transpose:  # 전치가 필요한 경우
            x = rearrange(x, 'b d n -> b n d')  # (B, D, N) -> (B, N, D)

        x = self.project_in(x)  # 입력을 코드북 차원으로 투영

        if is_multiheaded:  # 다중 헤드인 경우
            x = rearrange(x, 'b n (h d) -> (b h) n d', h=heads)  # (B, N, H*D) -> (B*H, N, D)

        quantize, embed_ind = self._codebook(x)  # 코드북 양자화 및 인덱스 추출

        if self.training:  # 학습 중인 경우
            quantize = x + (quantize - x).detach()  # 양자화된 값에 입력 값의 변화량을 추가

        if self.training:  # 학습 중인 경우
            if self.commitment_weight > 0:  # 커밋 손실 가중치가 0보다 큰 경우
                commit_loss = F.mse_loss(quantize.detach(), x)  # 커밋 손실 계산
                vq_loss['commit_loss'] = commit_loss  # 커밋 손실 저장
                vq_loss['loss'] = vq_loss['loss'] + commit_loss * self.commitment_weight  # 총 손실에 커밋 손실 추가

            if self.orthogonal_reg_weight > 0:  # 직교 정규화 가중치가 0보다 큰 경우
                codebook = self.codebook  # 코드북 벡터 추출

                if self.orthogonal_reg_active_codes_only:  # 활성 코드만 직교 손실 계산하는 경우
                    unique_code_ids = torch.unique(embed_ind)  # 활성 코드 인덱스 추출
                    codebook = codebook[unique_code_ids]  # 활성 코드만 코드북에서 추출

                num_codes = codebook.shape[0]  # 코드북 크기 저장
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:  # 최대 코드 개수가 지정되어 있고 코드 개수가 그보다 큰 경우
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]  # 랜덤으로 코드 인덱스 추출
                    codebook = codebook[rand_ids]  # 추출된 코드만 코드북에서 선택

                orthogonal_reg_loss = orthgonal_loss_fn(codebook)  # 직교 정규화 손실 계산
                vq_loss['orthogonal_reg_loss'] = orthogonal_reg_loss  # 직교 손실 저장
                vq_loss['loss'] = vq_loss['loss'] + orthogonal_reg_loss * self.orthogonal_reg_weight  # 총 손실에 직교 손실 추가

        if is_multiheaded:  # 다중 헤드인 경우
            quantize = rearrange(quantize, '(b h) n d -> b n (h d)', h=heads)  # (B*H, N, D) -> (B, N, H*D)
            embed_ind = rearrange(embed_ind, '(b h) n -> b n h', h=heads)  # (B*H, N) -> (B, N, H)

        quantize = self.project_out(quantize)  # 양자화된 값을 원래 차원으로 투영

        if need_transpose:  # 전치가 필요한 경우
            quantize = rearrange(quantize, 'b n d -> b d n')  # (B, N, D) -> (B, D, N)

        if self.accept_image_fmap:  # 입력이 이미지 피처 맵인 경우
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)  # (B, H*W, C) -> (B, C, H, W)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)  # (B, H*W, ...) -> (B, H, W, ...)

        return quantize, embed_ind, vq_loss, self._codebook.perplexity  # 양자화된 값, 인덱스, 손실, 퍼플렉서티 반환

if __name__ == '__main__':  # 메인 함수
    torch.manual_seed(0)  # 난수 시드 고정

    B, N, D = 1024, 32, 128  # 배치 크기, 시퀀스 길이, 차원 설정
    x = torch.rand((B, N, D))  # 무작위 입력 데이터 생성

    vq = VectorQuantize(dim=D, codebook_size=512)  # VectorQuantize 클래스 인스턴스 생성

    quantize, vq_ind, vq_loss, perplexity = vq(x)  # 입력 데이터를 양자화
    print(vq_ind[0])  # 첫 번째 인덱스 출력; 예: 87은 코드북에서 88번째 코드

    # 코드북 가중치를 가져올 수 있음
    print('vq.codebook.shape:', vq.codebook.shape)  # 코드북의 형태 출력; (코드북 크기, 차원)


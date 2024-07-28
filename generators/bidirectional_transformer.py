import math
from typing import Union

import torch
import torch.nn as nn
from einops import repeat
from x_transformers import ContinuousTransformerWrapper, Encoder as TFEncoder


def load_pretrained_tok_emb(pretrained_tok_emb, tok_emb, freeze_pretrained_tokens: bool):
    """
    사전 학습된 토큰 임베딩을 로드하고, 필요한 경우 동결합니다.
    
    :param pretrained_tok_emb: 사전 학습된 토큰 임베딩
    :param tok_emb: 현재 Transformer의 토큰 임베딩
    :param freeze_pretrained_tokens: 사전 학습된 토큰을 동결할지 여부
    """
    with torch.no_grad():
        if pretrained_tok_emb is not None:
            tok_emb.weight[:-1, :] = pretrained_tok_emb
            if freeze_pretrained_tokens:
                tok_emb.weight[:-1, :].requires_grad = False


class BidirectionalTransformer(nn.Module):
    def __init__(self,
                 kind: str,
                 num_tokens: int,
                 codebook_sizes: dict,
                 embed_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 heads: int,
                 ff_mult: int,
                 use_rmsnorm: bool,
                 p_unconditional: float,
                 n_classes: int,
                 pretrained_tok_emb_l: nn.Parameter = None,
                 pretrained_tok_emb_h: nn.Parameter = None,
                 freeze_pretrained_tokens: bool = False,
                 num_tokens_l: int = None,
                 **kwargs):
        """
        BidirectionalTransformer 클래스 초기화 메소드.
        
        :param kind: 'LF' 또는 'HF' (저주파 또는 고주파)
        :param num_tokens: 토큰 수
        :param codebook_sizes: 코드북 크기 딕셔너리
        :param embed_dim: 임베딩 차원
        :param hidden_dim: 은닉층 차원
        :param n_layers: 레이어 수
        :param heads: 헤드 수
        :param ff_mult: FF 곱
        :param use_rmsnorm: RMSNorm 사용 여부
        :param p_unconditional: 무조건적인 샘플링 확률
        :param n_classes: 클래스 수
        :param pretrained_tok_emb_l: 사전 학습된 저주파 토큰 임베딩
        :param pretrained_tok_emb_h: 사전 학습된 고주파 토큰 임베딩
        :param freeze_pretrained_tokens: 사전 학습된 토큰 동결 여부
        :param num_tokens_l: 저주파 토큰 수
        :param kwargs: 추가 인자
        """
        super().__init__()
        assert kind in ['LF', 'HF']
        self.kind = kind
        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim if kind == 'LF' else 2 * embed_dim
        out_dim = embed_dim

        # 토큰 임베딩
        self.tok_emb_l = nn.Embedding(codebook_sizes['lf'] + 1, embed_dim)  # `+1`은 마스크 토큰을 위한 것
        load_pretrained_tok_emb(pretrained_tok_emb_l, self.tok_emb_l, freeze_pretrained_tokens)
        if kind == 'HF':
            self.tok_emb_h = nn.Embedding(codebook_sizes['hf'] + 1, embed_dim)  # `+1`은 마스크 토큰을 위한 것
            load_pretrained_tok_emb(pretrained_tok_emb_h, self.tok_emb_h, freeze_pretrained_tokens)

        # Transformer 설정
        self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)  # `+1`은 무조건 조건을 위한 것
        self.blocks = ContinuousTransformerWrapper(dim_in=in_dim,
                                                   dim_out=in_dim,
                                                   max_seq_len=self.num_tokens + 1,
                                                   attn_layers=TFEncoder(
                                                       dim=hidden_dim,
                                                       depth=n_layers,
                                                       heads=heads,
                                                       use_rmsnorm=use_rmsnorm,
                                                       ff_mult=ff_mult,
                                                       use_abs_pos_emb=False,
                                                   ))
        self.Token_Prediction = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim, eps=1e-12)
        )
        codebook_size = codebook_sizes['lf'] if kind == 'LF' else codebook_sizes['hf']
        self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size + 1))
        self.ln = nn.LayerNorm(in_dim, eps=1e-12)
        self.drop = nn.Dropout(p=0.)

        if kind == 'HF':
            self.projector = nn.Conv1d(num_tokens_l, self.num_tokens, kernel_size=1)

    def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
        """
        클래스 임베딩을 생성합니다.
        
        :param class_condition: 클래스 조건 (없을 경우 무조건 샘플링)
        :param batch_size: 배치 크기
        :param device: 장치 (CPU/GPU)
        :return: 클래스 임베딩
        """
        if isinstance(class_condition, torch.Tensor):
            # 조건이 주어진 경우 (조건부 샘플링)
            conditional_ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            class_condition = torch.where(conditional_ind, class_condition.long(), class_uncondition)  # (b 1)
        else:
            # 조건이 주어지지 않은 경우 (무조건 샘플링)
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            class_condition = class_uncondition
        cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
        return cls_emb

    def forward_lf(self, embed_ind_l, class_condition: Union[None, torch.Tensor] = None):
        """
        저주파 데이터를 위한 순방향 전파 메소드.
        
        :param embed_ind_l: 저주파 임베딩 인덱스
        :param class_condition: 클래스 조건
        :return: 로짓
        """
        device = embed_ind_l.device

        token_embeddings = self.tok_emb_l(embed_ind_l)  # (b n dim)
        cls_emb = self.class_embedding(class_condition, embed_ind_l.shape[0], device)  # (b 1 dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = self.drop(self.ln(token_embeddings + position_embeddings))  # (b, n, dim)
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+n, dim)
        embed = self.blocks(embed)  # (b, 1+n, dim)
        embed = self.Token_Prediction(embed)[:, 1:, :]  # (b, n, dim)

        logits = torch.matmul(embed, self.tok_emb_l.weight.T) + self.bias  # (b, n, codebook_size+1)
        logits = logits[:, :, :-1]  # 마스크 토큰에 대한 로짓을 제거합니다.  # (b, n, codebook_size)
        return logits

    def forward_hf(self, embed_ind_l, embed_ind_h, class_condition=None):
        """
        고주파 데이터를 위한 순방향 전파 메소드.
        
        :param embed_ind_l: 저주파 임베딩 인덱스
        :param embed_ind_h: 고주파 임베딩 인덱스
        :param class_condition: 클래스 조건
        :return: 로짓
        """
        device = embed_ind_l.device

        token_embeddings_l = self.tok_emb_l(embed_ind_l)  # (b n dim)
        token_embeddings_l = self.projector(token_embeddings_l)  # (b m dim)
        token_embeddings_h = self.tok_emb_h(embed_ind_h)  # (b m dim)
        token_embeddings = torch.cat((token_embeddings_l, token_embeddings_h), dim=-1)  # (b m 2*dim)
        cls_emb = self.class_embedding(class_condition, embed_ind_l.shape[0], device)  # (b 1 2*dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = self.drop(self.ln(token_embeddings + position_embeddings))  # (b, m, 2*dim)
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+m, 2*dim)
        embed = self.blocks(embed)  # (b, 1+m, 2*dim)
        embed = self.Token_Prediction(embed)[:, 1:, :]  # (b, m, dim)

        logits = torch.matmul(embed, self.tok_emb_h.weight.T) + self.bias  # (b, m, codebook_size+1)
        logits = logits[:, :, :-1]  # 마스크 토큰에 대한 로짓을 제거합니다.  # (b, m, codebook_size)
        return logits

    def forward(self, embed_ind_l, embed_ind_h=None, class_condition: Union[None, torch.Tensor] = None):
        """
        순방향 전파 메소드.
        
        :param embed_ind_l: 저주파 임베딩 인덱스
        :param embed_ind_h: 고주파 임베딩 인덱스 (저주파 모델의 경우 None)
        :param class_condition: 클래스 조건 (없을 경우 무조건 샘플링 수행)
        :return: 로짓
        """
        if self.kind == 'LF':
            logits = self.forward_lf(embed_ind_l, class_condition)
        elif self.kind == 'HF':
            logits = self.forward_hf(embed_ind_l, embed_ind_h, class_condition)
        else:
            raise ValueError("kind는 'LF' 또는 'HF'이어야 합니다.")
        return logits
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.common_utils import get_n_previous_trading_date_from_year


class StockRankingTimeSeriesDataset(Dataset):
    def __init__(self, grouped_data):
        # grouped_data: dict[date] -> list of (x, y, industry_code, meta_info)
        self.data_by_date = list(grouped_data.items())

    def __len__(self):
        return len(self.data_by_date)

    def __getitem__(self, idx):
        date, daily_data = self.data_by_date[idx]
        return daily_data  # 텐서 변환 X (DataLoader에서 변환)


def get_stock_ranking_dataset(df, feature_columns, label_column, start_year, end_year, sliding_window_size, daily_topn_stocks, is_train):
    # 30일간 데이터를 확보하기 위해 시작 날짜 조정
    start_date = get_n_previous_trading_date_from_year(start_year, 30)
    end_date = f'{end_year}1231'
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # (임시 저장소) 날짜별 데이터 모음
    temp_grouped = defaultdict(list)

    # 종목별 슬라이딩 윈도우 생성
    df = df.sort_values(['stock_code', 'date'])
    for _, df_stock in df.groupby("stock_code"):
        features = df_stock[feature_columns].copy()
        for i in range(len(features) - sliding_window_size + 1):
            meta_info = df_stock.iloc[i + sliding_window_size - 1]
            if meta_info['year'] < start_year or meta_info['year'] > end_year:
                continue

            # 시총, 거래 대금, 상한가
            if meta_info['rank'] > daily_topn_stocks or meta_info['close_rate'] >= 29:
                continue

            x = features.iloc[i:i + sliding_window_size]  # (30, F)
            
            y = meta_info[label_column] / 100
            industry_id = meta_info['industry_id']  # 업종
            date = meta_info['date']

            temp_grouped[date].append((x.values, y, industry_id, meta_info))

    # 분위수 라벨 처리
    grouped_by_date = defaultdict(list)
    for date, group in temp_grouped.items():
        # 해당 날짜의 수익률 리스트
        returns = [sample[1] for sample in group]
        quantiles = pd.qcut(returns, q=20, labels=False)

        for i, (x, _, industry_id, meta_info) in enumerate(group):
            # label = int(quantiles[i])  # 분위수 기반 라벨
            grouped_by_date[date].append((x, quantiles[i], industry_id, meta_info))

    return StockRankingTimeSeriesDataset(grouped_by_date)


def get_stock_ranking_live_dataset(df, feature_columns):
    grouped_by_date = defaultdict(list)

    # 종목별 슬라이딩 윈도우 생성
    df = df.sort_values(['stock_code', 'date'])
    for _, df_stock in df.groupby("stock_code"):
        features = df_stock[feature_columns].copy()
        meta_info = df_stock.iloc[-1]

        # 상한가 근처 제외
        if meta_info['close_rate'] >= 29:
            continue

        x = features  # (30, F)
        y = 0.0
        industry_id = meta_info['industry_id']  # 업종
        date = meta_info['date']
        grouped_by_date[date].append((x.values, y, industry_id, meta_info))
    
    return StockRankingTimeSeriesDataset(grouped_by_date)

        
class StockPatchTST(nn.Module):
    def __init__(self, input_dim, patch_len, stride, model_dim, num_heads, num_layers, output_dim, industry_vocab_size, industry_embed_dim, dropout=0.0):
        super(StockPatchTST, self).__init__()

        # num_patches = ((seq_len - patch_len) / stride) + 1
        self.patch_len = patch_len  # 시계열을 의미 있는 단위로 분할(예: 주간 패턴)
        self.stride = stride  # 적절한 중복으로 패치 수 증가 (예: 30일 사이즈 윈도우를 6일 사이즈 패치로 분할하는데 stride가 3이면 총 9개의 패치가 만들어진다.)

        # 산업군 임베딩
        self.industry_embedding = nn.Embedding(industry_vocab_size, industry_embed_dim)

        self.total_input_dim = industry_embed_dim + input_dim  # concat 이후 총 feature 차원

        # Patch Embedding (Conv1D)
        self.patch_embedding = nn.Conv1d(
            in_channels=self.total_input_dim,
            out_channels=model_dim,
            kernel_size=patch_len,
            stride=stride
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully Connected Output
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x, industry_ids):  # x: (batch_size, seq_len, input_dim)
        # 산업군 임베딩 추가
        industry_embed = self.industry_embedding(industry_ids)  # (batch_size, embed_dim)
        industry_embed = industry_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([industry_embed, x], dim=-1)   # (batch, seq_len, total_input_dim)

        # Patch Embedding (Conv1D)
        x = x.permute(0, 2, 1)  # → (batch, total_input_dim, seq_len)
        x = self.patch_embedding(x)  # -> (batch, model_dim, num_patches)
        x = x.permute(0, 2, 1)  # → (batch, num_patches, model_dim)

        # Transformer Encoding
        x = self.transformer_encoder(x)  # (batch, num_patches, model_dim)

        # 평균 Pooling
        x = x.mean(dim=1)
        # 마지막 패치 (예측하려는 시점이 가장 최근 패치 직후에 있을 때 사용)
        # x = x[:, -1, :]  # shape: (batch_size, model_dim)

        # Fully Connected Output
        out = self.fc_out(x)  # shape: (batch_size, output_dim)
        return out


class LambdaRankLoss(nn.Module):
    def __init__(self, eps=1e-10, min_dcg=1e-6):
        super().__init__()
        self.eps = eps
        self.min_dcg = min_dcg

    def forward(self, preds, targets):
        """
        preds: (N,) 예측 점수
        targets: (N,) 실제 relevance
        """
        N = preds.size(0)
        device = preds.device

        # 1. relevance를 양수로 제한 (NDCG 정의 상)
        safe_targets = torch.clamp_min(targets, 0.0)

        # 2. ideal DCG 계산 (정답 순서대로 정렬)
        sorted_targets, _ = torch.sort(safe_targets, descending=True)
        ideal_dcg = self.dcg(sorted_targets)
        ideal_dcg = torch.clamp(ideal_dcg, min=self.min_dcg)  # 너무 작은 DCG 방지

        # 3. pairwise 차이 계산
        pred_diffs = preds.unsqueeze(1) - preds.unsqueeze(0)       # (N, N)
        target_diffs = targets.unsqueeze(1) - targets.unsqueeze(0) # (N, N)

        S_ij = torch.sign(target_diffs)  # 순서 비교 (+1, -1, 0)

        # 4. 예측값 순위 → 정확한 랭킹 계산
        rank_positions = self.compute_ranks(preds) - 1  # 0-based
        pos_i = rank_positions.unsqueeze(1)
        pos_j = rank_positions.unsqueeze(0)

        log_i = torch.log2(pos_i + 2.0)
        log_j = torch.log2(pos_j + 2.0)

        # 5. delta NDCG 계산 (쌍별 DCG 변화량 정규화)
        delta_dcg = torch.abs(1.0 / log_i - 1.0 / log_j) * torch.abs(target_diffs)
        delta_ndcg = delta_dcg / ideal_dcg
        delta_ndcg = torch.clamp(delta_ndcg, min=1e-4)  # 너무 작은 weight 방지

        # 6. loss 계산 (logsigmoid * delta_ndcg)
        log_loss = F.logsigmoid(S_ij * pred_diffs)

        # 유효한 쌍만 계산 (같은 타겟일 경우 제외)
        mask = (S_ij != 0).float()
        loss_matrix = -log_loss * delta_ndcg * mask

        loss_sum = loss_matrix.sum()
        pair_count = mask.sum()

        return loss_sum / pair_count if pair_count > 0 else torch.tensor(0.0, requires_grad=True, device=device)

    def dcg(self, relevance):
        device = relevance.device
        denom = torch.log2(torch.arange(relevance.size(0), device=device).float() + 2.0)
        return (relevance / denom).sum()

    def compute_ranks(self, scores):
        scores_np = scores.detach().cpu().numpy()
        ranks = scipy.stats.rankdata(-scores_np, method="ordinal")
        return torch.tensor(ranks, dtype=torch.float32, device=scores.device)


def stock_ranking_collate_fn(batch):
    daily_data = batch[0]  # batch_size=1이므로

    # numpy array 변환 후 tensor
    x_batch = torch.tensor(np.array([item[0] for item in daily_data]), dtype=torch.float32)
    y_batch = torch.tensor(np.array([item[1] for item in daily_data]), dtype=torch.float32)

    # industry_id 추출하여 텐서로 변환
    industry_id_batch = torch.tensor([item[2] for item in daily_data], dtype=torch.long)

    # meta_info는 변형 없이 리스트 유지
    meta_batch = [item[3] for item in daily_data]

    return x_batch, y_batch, industry_id_batch, meta_batch

    
# 랜덤 시드 설정
def set_seed(seed):    
    # Python 및 NumPy 랜덤 시드 고정
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch 랜덤 시드 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시 모든 GPU에 적용

    # PyTorch 연산의 결정론적 실행 보장
    torch.backends.cudnn.deterministic = True # 결정론적 연산 유지
    torch.backends.cudnn.benchmark = False  # 일관된 연산을 보장 (성능보다 재현성을 우선)


def seed_worker(worker_id):
    # 각 worker의 시드 고정
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



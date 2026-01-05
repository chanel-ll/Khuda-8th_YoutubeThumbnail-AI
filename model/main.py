# ==============================================================================
# 0. S E T U P
# ==============================================================================
# Google Colab 환경에서 실행하기 전, 아래 주석을 해제하여 필요한 라이브러리를 설치하세요.
# !pip install timm pandas openpyxl scikit-learn opencv-python
# !pip install koreanize-matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib # Matplotlib에서 한글 폰트 깨짐 방지
from PIL import Image
from tqdm.auto import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import seaborn as sns

# 재현성을 위한 랜덤 시드 고정
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ==============================================================================
# 1. D A T A   P R E P A R A T I O N
# ==============================================================================
# 이 섹션에서는 엑셀 파일 로드, 데이터 정제, 레이블 생성,
# 그리고 PyTorch Dataset 및 DataLoader 생성을 진행합니다.

def load_and_preprocess_data(excel_path, image_dir):
    """
    엑셀 파일을 로드하고, 점수 기반으로 레이블을 생성하며,
    학습에 사용할 데이터프레임을 준비합니다.

    Args:
        excel_path (str): 메타데이터가 포함된 엑셀 파일 경로
        image_dir (str): 이미지 파일이 저장된 디렉토리

    Returns:
        pd.DataFrame: 전처리가 완료된 데이터프레임
    """
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"에러: {excel_path} 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return None
    
    # 'videoid' 컬럼이 실제 이미지 파일명과 확장자 없이 일치한다고 가정합니다.
    # 예: videoid='_0Rlj3_w0hg', 이미지 파일='_0Rlj3_w0hg.jpg'
    # 만약 컬럼명이 다르다면 아래 'videoid'를 실제 컬럼명으로 수정해야 합니다.
    if '썸네일ID' not in df.columns or 'Thumbnail_Score' not in df.columns:
        print("에러: '썸네일ID' 또는 'Thumbnail_Score' 컬럼이 엑셀 파일에 없습니다.")
        return None

    # 점수 기준으로 상위/하위 30% 분위수 기준 임계값 계산 - 명확한 클래스 분리
    lower_quantile = df['Thumbnail_Score'].quantile(0.3)
    upper_quantile = df['Thumbnail_Score'].quantile(0.7)

    # 레이블 생성: Good(1), Bad(0)
    df['label'] = -1 # 기본값
    df.loc[df['Thumbnail_Score'] >= upper_quantile, 'label'] = 1 # Good
    df.loc[df['Thumbnail_Score'] <= lower_quantile, 'label'] = 0 # Bad

    # 레이블이 할당된 데이터만 필터링
    df_filtered = df[df['label'].isin([0, 1])].copy()

    # 이미지 파일 경로 생성
    df_filtered['file_path'] = df_filtered['썸네일ID'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

    # 메타데이터 추출 및 정규화 (색감, 텍스트비율)
    meta_cols = ['색감', '텍스트비율']
    for col in meta_cols:
        if col in df_filtered.columns:
            # Min-Max Scaling을 통해 0~1 사이로 정규화
            c_min, c_max = df_filtered[col].min(), df_filtered[col].max()
            if c_max > c_min:
                df_filtered[col] = (df_filtered[col] - c_min) / (c_max - c_min)
            else:
                df_filtered[col] = 0.0
        else:
            print(f"경고: '{col}' 컬럼이 엑셀에 없습니다. 0으로 채웁니다.")
            df_filtered[col] = 0.0

    # 파일이 실제로 존재하는지 확인
    df_filtered['file_exists'] = df_filtered['file_path'].apply(os.path.exists)
    
    missing_files = df_filtered[~df_filtered['file_exists']]
    if not missing_files.empty:
        print(f"경고: {len(missing_files)}개의 이미지 파일을 찾을 수 없습니다. 해당 데이터는 제외됩니다.")

    df_final = df_filtered[df_filtered['file_exists']].reset_index(drop=True)

    # CPU 환경에서 신속한 검토를 위해 100개 데이터 샘플링
    # MAX_SAMPLES = 100
    # if len(df_final) > MAX_SAMPLES:
    #     print(f"참고: 신속한 검토를 위해 {len(df_final)}개 중 {MAX_SAMPLES}개 데이터를 샘플링합니다.")
    #     df_final = df_final.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

    print(f"데이터 준비 완료. Good: {len(df_final[df_final['label']==1])}개, Bad: {len(df_final[df_final['label']==0])}개")
    
    return df_final


class ThumbnailDataset(Dataset):
    """YouTube 썸네일 데이터셋을 위한 커스텀 PyTorch Dataset"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'file_path']
        label = self.df.loc[idx, 'label']
        meta = self.df.loc[idx, ['색감', '텍스트비율']].values.astype(np.float32)

        # 이미지 로드 및 RGB 변환
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(meta, dtype=torch.float32)


# MixUp 증강을 위한 유틸리티 함수
def mixup_data(x, y, alpha=0.2):
    """
    MixUp 증강을 적용합니다.
    두 이미지와 레이블을 혼합하여 일반화 성능을 향상시킵니다.
    
    Args:
        x: 입력 이미지 배치
        y: 레이블 배치
        alpha: MixUp 혼합 비율 (Beta 분포 파라미터)
    
    Returns:
        mixed_x: 혼합된 이미지
        y_a, y_b: 두 레이블
        lam: 혼합 비율
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp에 대한 혼합 손실 계산"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_data_loaders(config):
    """데이터 로더와 검증 데이터프레임, 변환 함수를 생성하고 반환합니다."""
    
    df = load_and_preprocess_data(config["excel_path"], config["image_dir"])
    if df is None:
        return None, None, None, None

    # 데이터셋을 Train, Validation으로 분할 (80/20)
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label'] # 레이블 비율 유지
    )
    val_df = val_df.reset_index(drop=True)

    # timm 모델이 요구하는 표준 이미지 변환 (강화된 증강)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((config["image_size"] + 32, config["image_size"] + 32)),
            transforms.RandomCrop(config["image_size"]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),  # 회색조 변환 추가
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 블러 추가
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 어파인 변환 추가
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # Random Erasing 추가
        ]),
        'val': transforms.Compose([
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = ThumbnailDataset(train_df.reset_index(drop=True), transform=data_transforms['train'])
    val_dataset = ThumbnailDataset(val_df, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    print(f"Train 데이터셋: {len(train_dataset)}개, Val 데이터셋: {len(val_dataset)}개")
    return train_loader, val_loader, val_df, data_transforms['val']



# ==============================================================================
# 2. L O S S   F U N C T I O N S
# ==============================================================================
# 이 섹션에서는 MMD Loss와 Knowledge Distillation Loss를 구현합니다.

def gaussian_kernel(x, y, sigmas):
    """
    가우시안 커널을 계산합니다. 각 시그마 값에 대해 커널을 계산한 후 합산합니다.
    
    Args:
        x (Tensor): 첫 번째 텐서 (Batch, Dim)
        y (Tensor): 두 번째 텐서 (Batch, Dim)
        sigmas (list): 가우시안 커널에 사용할 시그마 값들의 리스트

    Returns:
        Tensor: 계산된 커널 값
    """
    # pairwise squared Euclidean distance (Batch, Batch)
    dist = torch.cdist(x, y, p=2.0) ** 2

    kernel_values = []
    for sigma in sigmas:
        beta = 1. / (2. * (sigma ** 2))
        kernel_values.append(torch.exp(-beta * dist))
    
    # 각 시그마에 대한 커널 값을 합산
    return torch.stack(kernel_values, dim=0).sum(dim=0)


def mmd_loss(x, y, sigmas=[1, 5, 10]):
    """
    Maximum Mean Discrepancy (MMD) 손실을 계산합니다.
    두 분포(x와 y)의 유사성을 측정하는 데 사용됩니다.

    Args:
        x (Tensor): 첫 번째 샘플 그룹 (Batch, Dim)
        y (Tensor): 두 번째 샘플 그룹 (Batch, Dim)
        sigmas (list): 가우시안 커널 시그마 값 리스트

    Returns:
        Tensor: MMD 손실 값
    """
    sigmas = torch.tensor(sigmas, device=x.device)
    
    # x와 x', y와 y', x와 y' 간의 커널 값 계산
    k_xx = gaussian_kernel(x, x, sigmas).mean()
    k_yy = gaussian_kernel(y, y, sigmas).mean()
    k_xy = gaussian_kernel(x, y, sigmas).mean()
    
    return k_xx + k_yy - 2 * k_xy


def kd_loss(student_logits, teacher_logits, temperature):
    """
    Knowledge Distillation (KD) 손실을 계산합니다.
    Student 모델(해석기)이 Teacher 모델(예측기)의 출력을 모방하도록 학습시킵니다.

    Args:
        student_logits (Tensor): Student 모델의 로짓
        teacher_logits (Tensor): Teacher 모델의 로짓
        temperature (float): 로짓을 부드럽게 만들기 위한 온도 파라미터

    Returns:
        Tensor: KD 손실 값
    """
    # Teacher의 로짓은 detach()하여 그래디언트가 전파되지 않도록 함
    soft_teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Student의 로짓에 log_softmax 적용
    soft_student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL-Divergence Loss 계산
    # reduction='batchmean'은 배치 크기로 평균을 내는 것을 의미합니다.
    loss = F.kl_div(soft_student_log_probs, soft_teacher_probs.detach(), reduction='batchmean')
    
    # 온도에 따라 스케일링
    loss = loss * (temperature ** 2)
    
    return loss


# ==============================================================================
# 3. M O D E L   A R C H I T E C T U R E
# ==============================================================================
# 이 섹션에서는 Single-Head Self-Attention 메커니즘과
# 전체 IA-ViT 모델을 정의합니다.

class SingleHeadSelfAttention(nn.Module):
    """
    해석기(Interpreter)를 위한 단일 헤드 자기 주의(Single-Head Self-Attention) 모듈.
    이미지 패치 임베딩을 입력받아 어텐션 가중치와 가중 평균된 임베딩을 출력합니다.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Q, K, V를 생성하기 위한 단일 선형 레이어
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.scale = embed_dim ** -0.5

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.3)  # 강화된 Dropout

    def forward(self, x):
        B, N, C = x.shape
        # (B, N, C) -> (B, N, 3 * C) -> (B, N, 3, C) -> (3, B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 어텐션 스코어 계산 (B, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn.softmax(dim=-1) # α^I (Interpreter attention)

        # 어텐션 스코어를 값에 적용 (B, N, C)
        x = (attn_probs @ v)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x, attn_probs

# timm의 Attention 모듈을 래핑하여 어텐션 가중치를 추출하기 위한 Wrapper
from timm.models.vision_transformer import Attention as TimmAttention

class AttentionWrapper(nn.Module):
    def __init__(self, original_attn_module):
        super().__init__()
        self.original_attn = original_attn_module
        self.attn_probs = None # 어텐션 가중치를 저장할 속성

    def forward(self, x, attn_mask=None):
        # original_attn 모듈의 내부 forward 로직을 모방하여 attn_probs를 추출
        B, N, C = x.shape
        # original_attn의 qkv를 사용하여 Q, K, V 계산
        qkv = self.original_attn.qkv(x).reshape(B, N, 3, self.original_attn.num_heads, C // self.original_attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 어텐션 스코어 계산
        attn = (q @ k.transpose(-2, -1)) * self.original_attn.scale
        
        if attn_mask is not None: # attn_mask가 존재하면 적용
            attn = attn + attn_mask

        # softmax 적용 및 어텐션 가중치 저장
        self.attn_probs = attn.softmax(dim=-1)

        # dropout 및 최종 출력 계산
        attn = self.original_attn.attn_drop(self.attn_probs)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.original_attn.proj(x)
        x = self.original_attn.proj_drop(x)
        return x


class IAViT(nn.Module):
    """
    Interpretability-Aware Vision Transformer (IA-ViT) 모델.
    - 특징 추출기(h): ViT 백본
    - 예측기(f): 클래스 토큰을 사용해 예측
    - 해석기(g): 이미지 패치 토큰을 사용해 예측기의 행동을 모사
    """
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. 특징 추출기(h) - ViT 백본 로드
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.embed_dim = self.backbone.embed_dim

        # --- 전이 학습을 위한 백본 파라미터 동결 (Gradual Unfreezing) ---
        # 먼저 모든 파라미터 동결
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 마지막 2개 블록만 학습 가능하게 설정 (도메인 적응)
        for block in self.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # patch_embed의 projection도 학습 가능하게 (선택적)
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
        # -----------------------------------------

        # 원본 ViT의 분류 헤드를 제거
        self.backbone.head = nn.Identity()

        # AttentionWrapper를 사용하여 ViT 마지막 블록의 어텐션 모듈 교체
        original_attn = self.backbone.blocks[-1].attn
        self.backbone.blocks[-1].attn = AttentionWrapper(original_attn)

        # --- Grad-CAM을 위한 Hook 설정 ---
        self.gradients = None
        self.activations = None
        
        def save_gradient(grad):
            self.gradients = grad

        def save_activation(module, input, output):
            # output is (B, N, C) for ViT blocks
            self.activations = output

        # 마지막 블록의 출력(norm)에 hook 설치
        self.backbone.norm.register_forward_hook(save_activation)
        self.backbone.norm.register_full_backward_hook(lambda module, grad_in, grad_out: save_gradient(grad_out[0]))
        # ---------------------------------

        # 2. 예측기 (f) - 이미지 피처 + 메타데이터 결합 (Late Fusion) - 강화된 구조
        self.meta_dim = 2 # 색감, 텍스트비율
        self.predictor = nn.Sequential(
            nn.Linear(self.embed_dim + self.meta_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),  # 강화된 Dropout
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # 3. 해석기 (g) (이 부분만 학습됨)
        self.interpreter_ssa = SingleHeadSelfAttention(self.embed_dim)
        self.interpreter_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x, features=None, meta_data=None):
        if x is not None:
            B = x.shape[0]
        elif features is not None:
            B = features.shape[0]
        elif meta_data is not None:
            B = meta_data.shape[0]
        else:
            B = 0
        
        # --- 특징 추출 (h) ---
        if features is not None:
            all_tokens = features
        else:
            # Grad-CAM 등을 위해 x에 gradient가 필요한 경우와 아닌 경우를 구분
            if x.requires_grad:
                all_tokens = self.backbone.forward_features(x)
            else:
                with torch.no_grad():
                    all_tokens = self.backbone.forward_features(x)
        
        cls_token = all_tokens[:, 0]
        patch_tokens = all_tokens[:, 1:]
        
        # --- 예측기 (f) ---
        # Late Fusion: cls_token과 meta_data 결합
        if meta_data is not None:
            combined_features = torch.cat([cls_token, meta_data], dim=1)
            predictor_logits = self.predictor(combined_features)
        else:
            # 메타데이터가 없는 예외 상황을 위해 (추론 시 등)
            dummy_meta = torch.zeros(B, self.meta_dim).to(cls_token.device)
            combined_features = torch.cat([cls_token, dummy_meta], dim=1)
            predictor_logits = self.predictor(combined_features)
        
        # --- 해석기 (g) ---
        interp_emb, interp_attn = self.interpreter_ssa(patch_tokens)
        interp_summary = interp_emb.mean(dim=1)
        interpreter_logits = self.interpreter_head(interp_summary)

        # --- 어텐션 정규화 (L_reg)를 위한 가중치 추출 ---
        interpreter_attention_to_reg = interp_attn.mean(dim=1)

        # AttentionWrapper에서 저장된 어텐션 가중치 사용
        extractor_attn_raw = self.backbone.blocks[-1].attn.attn_probs
        extractor_attention_to_reg = extractor_attn_raw[:, :, 0, 1:].mean(dim=1)

        interpreter_attention_to_reg = F.softmax(interpreter_attention_to_reg, dim=-1)
        extractor_attention_to_reg = F.softmax(extractor_attention_to_reg, dim=-1)

        return (
            predictor_logits,
            interpreter_logits,
            interpreter_attention_to_reg,
            extractor_attention_to_reg,
            interp_attn
        )


# ==============================================================================
# 4. T R A I N I N G   &   V A L I D A T I O N
# ==============================================================================
# 이 섹션에서는 모델 학습 및 검증을 위한 함수를 정의합니다.

def train_one_epoch(model, train_loader, optimizer, ce_criterion, config, device):
    """한 에폭(epoch) 동안 모델을 학습시킵니다."""
    model.train()
    total_loss, ce_loss_sum, kd_loss_sum, reg_loss_sum = 0, 0, 0, 0
    total_correct = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels, meta_data in progress_bar:
        images, labels, meta_data = images.to(device), labels.to(device), meta_data.to(device)

        # MixUp 증강 적용
        mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=config.get("mixup_alpha", 0.2))
        
        # Forward pass (features는 None, images를 전달하여 백본 통과)
        pred_logits, interp_logits, alpha_I, alpha_E, _ = model(mixed_images, features=None, meta_data=meta_data)
        
        # MixUp 손실 계산
        loss_ce = mixup_criterion(ce_criterion, pred_logits, labels_a, labels_b, lam)
        loss_kd = kd_loss(interp_logits, pred_logits, config["tau"])
        loss_reg = mmd_loss(alpha_I, alpha_E)
        
        # 개별 가중치로 손실 조합 (CE: 0.4, KD: 0.4, Reg: 0.2)
        loss = config["lambda_ce"] * loss_ce + config["lambda_kd"] * loss_kd + config["lambda_reg"] * loss_reg
        
        # Backward pass 및 파라미터 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계 기록
        total_loss += loss.item()
        ce_loss_sum += loss_ce.item()
        kd_loss_sum += loss_kd.item()
        reg_loss_sum += loss_reg.item()
        
        preds = pred_logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    avg_ce = ce_loss_sum / len(train_loader)
    avg_kd = kd_loss_sum / len(train_loader)
    avg_reg = reg_loss_sum / len(train_loader)
    accuracy = total_correct / len(train_loader.dataset)
    
    return avg_loss, avg_ce, avg_kd, avg_reg, accuracy


@torch.no_grad()
def validate_one_epoch(model, val_loader, ce_criterion, config, device):
    """한 에폭(epoch) 동안 모델의 성능을 검증합니다."""
    model.eval()
    total_loss, ce_loss_sum, kd_loss_sum, reg_loss_sum = 0, 0, 0, 0
    total_correct = 0

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    for images, labels, meta_data in progress_bar:
        images, labels, meta_data = images.to(device), labels.to(device), meta_data.to(device)

        # Forward pass (features는 None)
        pred_logits, interp_logits, alpha_I, alpha_E, _ = model(images, features=None, meta_data=meta_data)
        
        # 손실 계산
        loss_ce = ce_criterion(pred_logits, labels)
        loss_kd = kd_loss(interp_logits, pred_logits, config["tau"])
        loss_reg = mmd_loss(alpha_I, alpha_E)
        
        # 개별 가중치로 손실 조합 (CE: 0.4, KD: 0.4, Reg: 0.2)
        loss = config["lambda_ce"] * loss_ce + config["lambda_kd"] * loss_kd + config["lambda_reg"] * loss_reg
        
        # 통계 기록
        total_loss += loss.item()
        ce_loss_sum += loss_ce.item()
        kd_loss_sum += loss_kd.item()
        reg_loss_sum += loss_reg.item()
        
        preds = pred_logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(val_loader)
    avg_ce = ce_loss_sum / len(val_loader)
    avg_kd = kd_loss_sum / len(val_loader)
    avg_reg = reg_loss_sum / len(val_loader)
    accuracy = total_correct / len(val_loader.dataset)
    
    return avg_loss, avg_ce, avg_kd, avg_reg, accuracy



# ==============================================================================
# 5. V I S U A L I Z A T I O N
# ==============================================================================
# 이 섹션에서는 학습된 모델의 어텐션 맵을 시각화하는 함수를 정의합니다.
# 시각화를 위해 OpenCV가 필요합니다. (!pip install opencv-python)

def visualize_attention(model, image_path, transform, device, image_size=224):
    """
    학습된 모델의 해석기(Interpreter)가 생성한 어텐션 맵을 시각화합니다.

    Args:
        model (nn.Module): 학습된 IA-ViT 모델
        image_path (str): 시각화할 이미지의 경로
        transform (transforms.Compose): 이미지 전처리를 위한 변환 함수
        device (torch.device): 연산을 수행할 디바이스 (CPU 또는 CUDA)
        image_size (int): 모델 입력 이미지 크기
    """
    # 1. 이미지 로드 및 전처리
    original_img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    img_tensor = transform(original_img).unsqueeze(0).to(device)

    # 2. 모델 추론 및 어텐션 맵 추출
    model.eval()
    with torch.no_grad():
        _, _, _, _, interp_attn = model(img_tensor)
    
    # (1, N, N) -> (N, N), N = num_patches
    interp_attn = interp_attn.squeeze(0).cpu().numpy()

    # 3. 어텐션 맵 처리
    # 모든 패치에 대한 어텐션의 평균을 내어 각 패치의 중요도를 계산
    attn_map = interp_attn.mean(axis=0) # (N,)
    
    # 1D 어텐션 맵을 2D 그리드로 변환
    num_patches_side = int(np.sqrt(attn_map.shape[0]))
    attn_grid = attn_map.reshape(num_patches_side, num_patches_side)

    # 4. 히트맵 생성 및 오버레이
    # 2D 그리드를 원본 이미지 크기로 리사이즈
    attn_heatmap_resized = cv2.resize(attn_grid, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    
    # 정규화 및 컬러맵 적용
    attn_heatmap_normalized = (attn_heatmap_resized - np.min(attn_heatmap_resized)) / (np.max(attn_heatmap_resized) - np.min(attn_heatmap_resized))
    attn_heatmap_colored = (plt.cm.jet(attn_heatmap_normalized)[:, :, :3] * 255).astype(np.uint8)

    # 원본 이미지와 히트맵을 블렌딩
    original_img_np = np.array(original_img)
    overlayed_img = cv2.addWeighted(original_img_np, 0.6, attn_heatmap_colored, 0.4, 0)

    # 5. 결과 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(attn_heatmap_colored)
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlayed_img)
    axes[2].set_title('Overlayed Image')
    axes[2].axis('off')
    
    plt.suptitle(f"IA-ViT Attention Visualization for {os.path.basename(image_path)}", fontsize=16)
    plt.tight_layout()
    plt.savefig('attention_visualization.png') # 이미지를 파일로 저장
    plt.close() # 메모리에서 플롯을 닫음
    print("시각화 결과가 'attention_visualization.png' 파일로 저장되었습니다.")


@torch.no_grad()
def evaluate_and_visualize(model, val_loader, device):
    """
    전체 검증 데이터에 대해 모델 성능을 평가하고, 
    Classification Report, Confusion Matrix, ROC-AUC 곡선을 시각화합니다.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    print("\n[ 성능 평가 시작 ]")
    for images, labels, meta_data in tqdm(val_loader, desc="Evaluating"):
        images, labels, meta_data = images.to(device), labels.to(device), meta_data.to(device)
        
        # features와 meta_data가 이미 캐싱된 경우를 가정 (main driver 흐름 따름)
        pred_logits, _, _, _, _ = model(images, features=None, meta_data=meta_data)
        
        probs = F.softmax(pred_logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy()) # 클래스 1 (Good)에 대한 확률

    # 1. Classification Report 출력
    report = classification_report(all_labels, all_preds, target_names=['Bad', 'Good'])
    print("\n--- Classification Report ---")
    print(report)

    # 2. Confusion Matrix 시각화
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("성능 지표: Confusion Matrix가 'confusion_matrix.png'로 저장되었습니다.")

    # 3. ROC Curve 시각화
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png')
    plt.close()
    print(f"성능 지표: ROC 곡선이 'roc_curve.png'로 저장되었습니다. (AUC: {auc_score:.4f})")



# ==============================================================================
# 6. M A I N   D R I V E R
# ==============================================================================
# 이 섹션에서는 위에서 정의한 모든 구성 요소를 결합하여
# 전체 프로세스를 실행합니다.

def main():
    """메인 실행 함수"""
    # 하이퍼파라미터 및 설정 값 (개선됨)
    CONFIG = {
        "excel_path": "썸네일_분석결과_수정본.xlsx",
        "image_dir": "image/",
        "image_size": 224,
        "batch_size": 32,
        "epochs": 30,  # 충분한 학습을 위해 에폭 수 증가
        "learning_rate": 3e-5,  # 안정적인 학습을 위해 학습률 감소
        "warmup_epochs": 3,  # Warmup 에폭 수
        # 개별 손실 가중치 (beta 대신)
        "lambda_ce": 0.4,     # CE Loss 가중치
        "lambda_kd": 0.4,     # KD Loss 가중치  
        "lambda_reg": 0.2,    # Regularization Loss 가중치
        "tau": 2.0,           # KD Loss의 온도 파라미터
        "mixup_alpha": 0.2,   # MixUp 혼합 비율
        "model_save_path": "ia_vit_best.pth",
    }
    
    # 1. 데이터 로더 준비
    print("Step 1: 데이터 로딩 및 전처리를 시작합니다...")
    train_loader, val_loader, val_df, val_transform = get_data_loaders(CONFIG)
    if train_loader is None:
        print("데이터 로더 생성에 실패하여 프로그램을 종료합니다.")
        return
    print("="*60)

    # 2. 모델, 옵티마이저, 손실 함수 초기화
    print("Step 2: 모델, 옵티마이저, 손실 함수를 초기화합니다...")
    model = IAViT().to(DEVICE)
    
    # 학습할 파라미터만 필터링하여 옵티마이저에 전달
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG["learning_rate"], weight_decay=0.05)  # 강화된 Weight Decay
    
    # Warmup + Cosine Annealing 스케줄러
    def warmup_lambda(epoch):
        if epoch < CONFIG["warmup_epochs"]:
            return (epoch + 1) / CONFIG["warmup_epochs"]
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"] - CONFIG["warmup_epochs"], eta_min=1e-6
    )
    
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # 강화된 Label Smoothing
    print("모델의 백본은 동결되었으며, 새로 추가된 레이어만 학습합니다.")
    print("="*60)

    # 캐싱 로직 제거됨 - 실시간 증강을 위해
    
    # 3. 모델 학습
    print("Step 3: 모델 학습을 시작합니다...")
    best_val_loss = float('inf')

    # Early Stopping 설정
    patience = 5
    counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"--- Epoch {epoch}/{CONFIG['epochs']} ---")
        
        train_loss, tr_ce, tr_kd, tr_reg, train_acc = train_one_epoch(
            model, train_loader, optimizer, ce_criterion, CONFIG, DEVICE
        )
        # Warmup 또는 Cosine 스케줄러 업데이트
        if epoch <= CONFIG["warmup_epochs"]:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        val_loss, val_ce, val_kd, val_reg, val_acc = validate_one_epoch(
            model, val_loader, ce_criterion, CONFIG, DEVICE
        )
        
        print(f"  Train | Loss: {train_loss:.4f} (CE: {tr_ce:.4f}, KD: {tr_kd:.4f}, Reg: {tr_reg:.4f}), Acc: {train_acc:.4f}")
        print(f"  Valid | Loss: {val_loss:.4f} (CE: {val_ce:.4f}, KD: {val_kd:.4f}, Reg: {val_reg:.4f}), Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"  >> Best model saved to {CONFIG['model_save_path']} (Val Loss: {best_val_loss:.4f})")
            counter = 0
        else:
            counter += 1
            print(f"  Early Stopping Counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break
    print("="*60)

    # 4. 정밀 성능 평가 및 지표 시각화
    print("Step 4: 모델 성능 통합 평가를 시작합니다...")
    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=DEVICE))
    evaluate_and_visualize(model, val_loader, DEVICE)
    print("="*60)

    # 5. XAI 시각화
    print("Step 5: 학습된 모델로 XAI 분석 및 시각화를 시작합니다...")
    if not val_df.empty:
        model.eval()
        
        # 샘플 이미지 로드 (첫 번째 검증 데이터)
        sample_idx = 0
        sample_image_path = val_df.loc[sample_idx, 'file_path']
        label = val_df.loc[sample_idx, 'label']
        _, _, _, sample_meta = val_loader.dataset[sample_idx]
        
        print(f"시각화 및 분석 대상: {sample_image_path} (정답 레이블: {label})")
        
        
        # 데이터 준비
        img_pil = Image.open(sample_image_path).convert("RGB")
        img_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)
        
        # val_loader.dataset은 이제 Tuple (img, label, meta)를 리턴하므로
        # sample_meta를 가져오기 위해 직접 접근
        _, _, sample_meta_raw = val_loader.dataset[sample_idx]
        # Dataset에서 이미 tensor로 변환해서 주므로 그대로 사용하거나, 차원만 맞춤
        # 하지만 val_loader.dataset[sample_idx] 호출 시 transform이 적용되어 나오므로
        # 위에서 img_pil을 다시 로드해서 transform하는 것과 중복될 수 있음.
        # 여기서는 기존 코드 흐름을 유지하되 meta만 가져옴.
        
        meta_tensor = sample_meta_raw.unsqueeze(0).to(DEVICE)
        
        # 1. Grad-CAM 계산을 위해 gradient 활성화
        img_tensor.requires_grad = True 
        pred_logits, _, _, _, interp_attn = model(img_tensor, meta_data=meta_tensor)
        
        # 타겟 클래스 (예측된 클래스)에 대해 역전파
        target_class = pred_logits.argmax(dim=1).item()
        pred_logits[0, target_class].backward()
        
        # Hook에서 잡은 값들 (Backbone norm layer)
        grads = model.gradients # (1, 197, 768)
        acts = model.activations # (1, 197, 768)
        
        if grads is None or acts is None:
            print("경고: Grad-CAM을 위한 그래디언트 또는 액티베이션이 캡처되지 않았습니다.")
            # 더미 데이터로 대체하여 프로세스 계속 진행
            cam_np = np.zeros((196,))
        else:
            # 패치 토큰들에 대해서만 계산 (CLS 토큰 제외)
            grads = grads[:, 1:, :] 
            acts = acts[:, 1:, :]
            
            # Global Average Pooling of Gradients
            weights = torch.mean(grads, dim=1, keepdim=True) # (1, 1, 768)
            cam = torch.sum(weights * acts, dim=-1).squeeze(0) # (196,)
            cam = F.relu(cam) # ReLU 적용
            cam_np = cam.detach().cpu().numpy()
        
        # 2. IA-ViT Attention Map
        interp_attn_np = interp_attn.detach().squeeze(0).cpu().numpy().mean(axis=0) # (196,)
        
        # 3. 패치 기여도 수치화 (%)
        total_importance = cam_np + interp_attn_np
        normalized_importance = (total_importance / (total_importance.sum() + 1e-8)) * 100
        
        # 상위 5개 패치 추출
        top_k = 5
        top_indices = np.argsort(normalized_importance)[-top_k:][::-1]
        
        print("\n[ 패치 기여도 분석 결과 (Top 5) ]")
        grid_size = 14
        for i, idx in enumerate(top_indices):
            row = idx // grid_size
            col = idx % grid_size
            val = normalized_importance[idx]
            print(f"순위 {i+1}: 패치 위치 [{row}, {col}] - 기여도: {val:.2f}%")

        # 4. 시각화 (Triple-View)
        image_size = CONFIG["image_size"]
        original_img_np = np.array(img_pil.resize((image_size, image_size)))
        
        def get_heatmap(map_data):
            grid = map_data.reshape(grid_size, grid_size)
            resized = cv2.resize(grid, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            normed = (resized - np.min(resized)) / (np.max(resized) - np.min(resized) + 1e-8)
            colored = (plt.cm.jet(normed)[:, :, :3] * 255).astype(np.uint8)
            return colored

        ia_heatmap = get_heatmap(interp_attn_np)
        grad_heatmap = get_heatmap(cam_np)
        
        ia_overlay = cv2.addWeighted(original_img_np, 0.6, ia_heatmap, 0.4, 0)
        grad_overlay = cv2.addWeighted(original_img_np, 0.6, grad_heatmap, 0.4, 0)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(original_img_np); axes[0, 0].set_title('Original Image'); axes[0, 0].axis('off')
        axes[0, 1].imshow(ia_overlay); axes[0, 1].set_title('IA-ViT (Attention Intent)'); axes[0, 1].axis('off')
        axes[0, 2].imshow(grad_overlay); axes[0, 2].set_title('Grad-CAM (Numerical Contribution)'); axes[0, 2].axis('off')
        
        axes[1, 0].imshow(ia_heatmap); axes[1, 0].set_title('IA-ViT Heatmap'); axes[1, 0].axis('off')
        axes[1, 1].imshow(grad_heatmap); axes[1, 1].set_title('Grad-CAM Heatmap'); axes[1, 1].axis('off')
        
        # 분석 요약 텍스트
        summary_text = (f"Top Patch: [{top_indices[0]//14}, {top_indices[0]%14}]\n"
                        f"Contribution: {normalized_importance[top_indices[0]]:.1f}%\n"
                        f"Predicted Class: {['Bad', 'Good'][target_class]}")
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=15, verticalalignment='center')
        axes[1, 2].set_title('Analysis Summary'); axes[1, 2].axis('off')
        
        plt.suptitle("XAI Advanced Analysis: IA-ViT vs Grad-CAM", fontsize=20)
        plt.tight_layout()
        plt.savefig('attention_visualization_xai.png')
        plt.close()
        print("\n분석 결과가 'attention_visualization_xai.png' 파일로 저장되었습니다.")
    else:
        print("시각화할 검증 이미지가 없습니다.")
    print("="*60)
    print("프로세스가 성공적으로 완료되었습니다.")


if __name__ == '__main__':
    main()

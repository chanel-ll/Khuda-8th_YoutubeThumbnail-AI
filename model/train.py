import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import heapq
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import clip  # pip install git+https://github.com/openai/CLIP.git

class CLIPResNet50Regressor(nn.Module):
    def __init__(self, device='cuda'):
        super(CLIPResNet50Regressor, self).__init__()
        
        print(f"🚀 Loading OpenAI CLIP (RN50) model on {device}...")
        self.clip_model, _ = clip.load("RN50", device=device, jit=False)
        self.visual_encoder = self.clip_model.visual.float()
        
        visual_dim = 1024 
        intensity_dim = 1   
        
        self.regressor = nn.Sequential(
            nn.Linear(visual_dim + intensity_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def forward(self, images, intensities):
        visual_features = self.visual_encoder(images.float())
        combined = torch.cat((visual_features, intensities), dim=1)
        output = self.regressor(combined)
        return output

class MultiModalDataset(Dataset):
    def __init__(self, excel_path, img_dir):
        self.df = pd.read_excel(excel_path)
        self.df.columns = [col.strip() for col in self.df.columns]
        self.img_dir = img_dir
        
        if 'Emo_Intensity' not in self.df.columns:
            raise ValueError("❌ 엑셀에 'Emo_Intensity' 컬럼이 없습니다.")

        self.data = [] 
    
        EXTREME_LOW = 25.0; EXTREME_HIGH = 85.0  
        MODERATE_LOW = 35.0; MODERATE_HIGH = 75.0 
        
        original_count = 0
        augmented_count = 0
        
        print(f"데이터 로드 및 밸런싱(Lite Version) 작업 시작...")
        
        for _, row in self.df.iterrows():
            img_id = str(row['썸네일ID']).strip()
            
            # 이미지 파일 찾기
            found = False
            img_path = ""
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                temp_path = os.path.join(self.img_dir, f"{img_id}{ext}")
                if os.path.exists(temp_path):
                    img_path = temp_path
                    found = True
                    break
            
            if found:
                original_count += 1
                score = float(row['Thumbnail_Score'])
                intensity = float(row['Emo_Intensity'])
                
                data_item = {
                    'img_path': img_path,
                    'score': score,
                    'intensity': intensity
                }
                
                repeat_count = 1 

                if score <= EXTREME_LOW or score >= EXTREME_HIGH:
                    repeat_count = 3

                elif score <= MODERATE_LOW or score >= MODERATE_HIGH:
                    repeat_count = 2
                
                for _ in range(repeat_count):
                    self.data.append(data_item)
                    
                augmented_count += (repeat_count - 1)

        print("-" * 50)
        print(f"📊 [데이터셋 구축 완료]")
        print(f" - 원본 데이터 개수: {original_count}")
        print(f" - 증강된 데이터 개수: +{augmented_count}")
        print(f" - 최종 학습 데이터셋 크기: {len(self.data)}")
        print("-" * 50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            image = Image.open(item['img_path']).convert('RGB')
        except Exception:
            # 예외 처리: 검은 화면
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        label = torch.tensor(item['score'] / 100.0, dtype=torch.float32)
        intensity = torch.tensor([item['intensity']], dtype=torch.float32)
        
        return image, intensity, label

class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        img, intensity, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, intensity, label
        
    def __len__(self):
        return len(self.subset)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    excel_path = ""
    img_dir = ""

    _, clip_preprocess = clip.load("RN50", device=device, jit=False)

    train_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        ], p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        clip_preprocess 
    ])


    val_transform = clip_preprocess

    try:
        base_dataset = MultiModalDataset(excel_path, img_dir)
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return

    if len(base_dataset) == 0:
        print("❌ 학습할 데이터가 없습니다.")
        return

    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size])

    train_dataset = TransformedDataset(train_subset, transform=train_transform)
    val_dataset = TransformedDataset(val_subset, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    scores = [item['score'] for item in base_dataset.data]
    print(f"점수 표준편차(Std): {np.std(scores):.4f}")

    print(f"🚀 멀티모달 모델 학습 시작 (CLIP-ResNet50 + Emotion Intensity)... Device: {device}")
    model = CLIPResNet50Regressor(device=device).to(device)
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    top_k_models = [] 
    
    TOTAL_EPOCHS = 200
    patience_limit = 15
    patience_check = 0
    best_loss = float('inf')
    
    print(f"총 {TOTAL_EPOCHS} Epoch 학습 진행 (Early Stopping 적용)...")
    
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        train_loss = 0
        
        for imgs, intensities, labels in train_loader:
            imgs = imgs.to(device)
            intensities = intensities.to(device)
            labels = labels.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(imgs, intensities)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, intensities, labels in val_loader:
                imgs = imgs.to(device)
                intensities = intensities.to(device)
                labels = labels.to(device).view(-1, 1)
                
                outputs = model(imgs, intensities)
                
                val_loss += criterion(outputs, labels).item() * imgs.size(0)
                

                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(labels.cpu().numpy().flatten())

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        
        val_mae = mean_absolute_error(all_targets, all_preds)
        val_r2 = r2_score(all_targets, all_preds)
        
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

        save_path = f"model_clip_ep{epoch+1}_loss{val_loss:.4f}_mae{val_mae:.4f}.pth"
        
        if len(top_k_models) < 3:
            torch.save(model.state_dict(), save_path)
            heapq.heappush(top_k_models, (-val_loss, save_path)) 
            print(f"   💾 Top 3 진입! 모델 저장: {save_path}")
            
        else:
            worst_loss_neg, worst_path = top_k_models[0]
            worst_loss = -worst_loss_neg
            
            if val_loss < worst_loss:
                if os.path.exists(worst_path):
                    try:
                        os.remove(worst_path)
                        print(f"   🗑️ 순위 밀림 -> 파일 삭제: {worst_path}")
                    except OSError:
                        pass
                
                heapq.heappop(top_k_models)
                torch.save(model.state_dict(), save_path)
                heapq.heappush(top_k_models, (-val_loss, save_path))
                print(f"   💾 Top 3 갱신! 모델 저장: {save_path}")
                
        if val_loss < best_loss:
            best_loss = val_loss
            patience_check = 0 
        else:
            patience_check += 1
            print(f"   ⏳ 성능 갱신 없음 ({patience_check}/{patience_limit})")
            
            if patience_check >= patience_limit:
                print(f"\n🛑 Early Stopping 발동! (Epoch {epoch+1}에서 학습 종료)")
                break

    print("\n✅ 모든 학습 완료. 최종 저장된 Top 3 모델:")
    for loss_neg, path in sorted(top_k_models, key=lambda x: x[0], reverse=True): 
        print(f" - {path} (Loss: {-loss_neg:.4f})")

if __name__ == "__main__":
    train_model()
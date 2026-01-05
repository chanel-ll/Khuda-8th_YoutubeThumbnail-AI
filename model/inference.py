import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import pipeline
import clip  # pip install git+https://github.com/openai/CLIP.git

class CLIPResNet50Regressor(nn.Module):
    def __init__(self, device='cuda'):
        super(CLIPResNet50Regressor, self).__init__()
        # CLIP 로드 (JIT=False 필수)
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

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.current_intensity = None 
    
    def forward(self, x):
        return self.model(x, self.current_intensity)

class MultimodalPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP-ResNet50 Model on {self.device}...")

        _, self.clip_preprocess = clip.load("RN50", device=self.device, jit=False)
        
        original_model = CLIPResNet50Regressor(device=self.device).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        original_model.load_state_dict(checkpoint)
        original_model.eval()

        self.wrapper = ModelWrapper(original_model).to(self.device)
        
        self.target_layers = [self.wrapper.model.visual_encoder.layer4[-1]]
        self.cam = GradCAM(model=self.wrapper, target_layers=self.target_layers)
        
        print("Loading Sentiment Analysis Pipeline...")
        self.sentiment_analyzer = pipeline("zero-shot-image-classification", 
                                           model="openai/clip-vit-base-patch32", 
                                           device=0 if torch.cuda.is_available() else -1)

    def get_intensity(self, image_pil):
        candidate_labels = ["neutral", "positive", "negative"]
        try:
            results = self.sentiment_analyzer(image_pil, candidate_labels=candidate_labels)
            scores = {res['label']: res['score'] for res in results}
            intensity = 1.0 - scores.get('neutral', 0.0)
        except Exception:
            intensity = 0.5 
        return intensity

    def predict(self, image_path, save_heatmap=True):
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil.resize((224, 224))) / 255.0
        
        img_tensor = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
        
        intensity_val = self.get_intensity(img_pil)
        intensity_tensor = torch.tensor([[intensity_val]], dtype=torch.float32).to(self.device)
        
        print(f"📊 감정 강도(Intensity): {intensity_val:.4f}")
        self.wrapper.current_intensity = intensity_tensor

        with torch.no_grad():
            score = self.wrapper(img_tensor).item()

        targets = [ClassifierOutputTarget(0)]
        
        try:
            grayscale_cam = self.cam(input_tensor=img_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            if save_heatmap:
                final_score = score * 100
                save_name = f"clip_result_{final_score:.2f}_inten{intensity_val:.2f}.jpg"
                cv2.imwrite(save_name, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                print(f"📸 결과 저장: {save_name}")
                
        except Exception as e:
            print(f"⚠️ Grad-CAM 생성 실패: {e}")
            visualization = None

        return score * 100, intensity_val

if __name__ == "__main__":

    MODEL_PATH = ""
    
    # [설정] 테스트할 이미지 경로
    TEST_IMAGE = ""
    
    # 실행
    import os
    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_IMAGE):
        predictor = MultimodalPredictor(MODEL_PATH)
        final_score, intensity = predictor.predict(TEST_IMAGE)
        print("-" * 30)
        print(f"🎯 CLIP-AI 예측 점수: {final_score:.2f}점")
        print("-" * 30)
    else:
        print("❌ 경로 오류: 모델 파일이나 이미지 파일을 찾을 수 없습니다.")
        print(f"모델: {MODEL_PATH}")
        print(f"이미지: {TEST_IMAGE}")
import google.generativeai as genai
from PIL import Image
import os

class GeminiFeedback:
    def __init__(self, api_key):
        genai.configure(api_key=api_key) 
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_advice(self, original_img_path, heatmap_img_path, score, intensity):
        """
        original_img_path: 원본 썸네일 경로
        heatmap_img_path: Grad-CAM 결과물 경로 (빨간색 히트맵)
        score: 예측 점수
        intensity: 감정 강도
        """
        
        img_original = Image.open(original_img_path)
        img_heatmap = Image.open(heatmap_img_path)

        prompt = f"""
        당신은 전문 유튜브 썸네일 컨설턴트입니다. 
        사용자가 제작한 썸네일과, AI 모델이 이 썸네일을 분석한 'Grad-CAM 히트맵(시선 추적)' 결과를 보고 피드백을 제공해주세요.

        [데이터 정보]
        1. AI 예측 클릭 점수: {score:.2f}점 
        2. 썸네일 감정 강도: {intensity:.2f} 
        
        [이미지 설명]
        - 첫 번째 이미지: 원본 썸네일입니다.
        - 두 번째 이미지: AI 모델의 시선(Attention)을 시각화한 히트맵입니다. 붉은색 영역이 AI가 가장 집중해서 본 곳입니다.

        [요청 사항]
        위 정보를 바탕으로 다음 내용을 포함한 3줄 요약 피드백을 한국어로 작성해주세요:
        1. **AI의 시선 분석:** AI가 썸네일의 어느 부분(인물, 텍스트, 배경 등)에 주목했는지, 혹은 엉뚱한 곳(배경, 구석 등)을 보았는지 분석하세요.
        2. **점수 원인 진단:** 점수가 {score:.2f}점으로 나온 이유를 시각적 요소(표정, 텍스트 가독성, 구도)와 연결해 설명하세요.
        3. **개선 제안:** 클릭률을 높이기 위해 구체적으로 무엇을 고쳐야 할지(예: "인물을 키우세요", "텍스트 색상을 바꾸세요") 조언하세요.
        
        말투는 전문가답지만 친절하게 부탁합니다.
        """

        try:
            response = self.model.generate_content([prompt, img_original, img_heatmap])
            return response.text
        except Exception as e:
            return f"❌ Gemini API 오류 발생: {e}"


if __name__ == "__main__":
    #구글 클라우드 콘솔에서 받은 API 키 입력
    MY_API_KEY = ""
    

    original_file = ""
    heatmap_file =" " 
    

    if os.path.exists(original_file) and os.path.exists(heatmap_file):
        advisor = GeminiFeedback(MY_API_KEY)
        
        print("🤖 Gemini가 썸네일을 분석 중입니다...")
        advice = advisor.generate_advice(
            original_file, 
            heatmap_file, 
            score=42.60, 
            intensity=1.0
        )
        
        print("\n" + "="*50)
        print("📢 [Gemini AI 컨설팅 결과]")
        print("="*50)
        print(advice)
    else:
        print("파일이 없습니다. 경로를 확인해주세요.")
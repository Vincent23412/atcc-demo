from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from deepface import DeepFace

app = FastAPI()

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 請求格式
class ExpressionRequest(BaseModel):
    expression: str
    image_base64: str

# 中文表情對應英文（DeepFace 回傳用英文）
expression_map = {
    "高興": "happy",
    "生氣": "angry",
    "悲傷": "sad",
    "驚訝": "surprise",
    "中性": "neutral",
    "害怕": "fear",
    "厭惡": "disgust"
}

@app.post("/analyze-expression")
def analyze_expression(req: ExpressionRequest):
    try:
        # 解碼 base64 -> 圖片陣列
        image_data = base64.b64decode(req.image_base64)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 使用 DeepFace 做情緒分析
        analysis = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion'].lower()

        # 對應中文輸入表情
        target_emotion = expression_map.get(req.expression.strip(), "").lower()
        is_match = (dominant_emotion == target_emotion)

        result_text = (
            f"分析結果：人物表情為「{dominant_emotion}」，"
            + ("與指定表情相符 ✅" if is_match else "與指定表情不相符 ❌")
        )
        return {"result": result_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepFace 分析錯誤：{str(e)}")

from vertexai import init
from vertexai.preview.generative_models import GenerativeModel
from dotenv import load_dotenv
import os

# 이미지 리사이징 
from PIL import Image as PILImage
from vertexai.preview.generative_models import Image as VertexImage

# GCP Cloud Storage 연결
from google.cloud import storage  
import tempfile                     # 임시 파일 저장용

# LangChain 적용
from langchain_google_vertexai import VertexAI
from model.verify.event_challenge_prompt import event_challenge_prompts


class ImageVerifyModel :
    def __init__(self, credential_env="GOOGLE_APPLICATION_CREDENTIALS", project_id="leafresh", region="us-central1"): 
        # 환경변수 로드 및 인증 초기화
        load_dotenv()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(credential_env)
        init(project=project_id, location=region)                                       # Vertex AI 프로젝트/리전 초기화
        self.model = GenerativeModel("gemini-2.0-flash")                                # 모델 정의
        self.storage_client = storage.Client()                                          # GCS 클라이언트 


    def image_verify(self, bucket_name: str, blob_name: str, challenge_type: str, challenge_id: int, challenge_name: str) -> str :
        try:
            bucket = self.storage_client.bucket(bucket_name)                            # 이미지 업로드 
            blob = bucket.blob(blob_name)                                 
       
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                blob.download_to_filename(temp_file.name) 

                # 이벤트 챌린지인 경우에만 리사이징 수행
                if challenge_type.upper() == "GROUP" and 1 <= challenge_id <= 17:
                    pillow_image = PILImage.open(temp_file.name).convert("RGB")
                    if max(pillow_image.size) > 1024:
                        new_width = 1024
                        new_height = int(pillow_image.height * 1024 / pillow_image.width)
                        pillow_image = pillow_image.resize((new_width, new_height))
                    pillow_image.save(temp_file.name, format="PNG")

                # VertexAI용 이미지 객체 로드
                image = VertexImage.load_from_file(temp_file.name)
                # image = Image.load_from_file(temp_file.name)

            return self.response(image, challenge_type, challenge_id, challenge_name)

        except Exception as e:
            return f"[에러] GCS 이미지 로드 실패: {e}" 


    def select_prompt(self, challenge_type: str, challenge_id: int):
        if challenge_type.upper() == "GROUP" and 1 <= challenge_id <= 17:
            return event_challenge_prompts.get(challenge_id)
        return None

    def response(self, image, challenge_type, challenge_id, challenge_name):
        prompt_template = self.select_prompt(challenge_type, challenge_id)

        if prompt_template:
            # LangChain 방식 (이벤트 챌린지)
            # llm = VertexAI(model_name="gemini-2.0-flash")
            # chain = prompt_template | llm
            # return chain.invoke({"image": image})
            prompt = prompt_template.format_prompt().to_string()
        else:
            prompt = (
                f"이 이미지는 '{challenge_name}'에 적합한 이미지 인가요? \n"
                "분위기가 아니라 물체가 존재해야합니다. 텀블러를 사용한 것이 맞으면 모두 '예'로 출력해주세요. \n"
                "고기를 제외하고 생선은 샐러드/채식 식단으로 모두 '예'를 출력해주세요. \n"
                "장바구니/에코백 챌린지의 경우 가방이 잘 나와있다면 모두 '예'를 출력해주세요. \n"
                "만보 걷기 챌린지 같은 경우 10000이상인 숫자가 있으면 '예'를 출력해주세요. \n"
                "작은 텃밭 가꾸기는 작은 화단의 모습이 나왔을 경우 '예'를 출력해주세요. \n"
                "너무 이미지가 흐리거나 블러 처리 되어있는 경우 무조건 '아니오'를 출력해주세요. \n"
                "적합한 이미지인지 예/아니오로 대답해주세요. 결과는 무조건 예/아니오 로만 대답해주세요. \n"
            )
            
        result = self.model.generate_content(
            [prompt, image],
            generation_config={
                "temperature": 0.4,
                "top_p": 1,
                "top_k": 32,
                "max_output_tokens": 512
            }
        )

        return result.text
    
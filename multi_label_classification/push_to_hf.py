import os
from huggingface_hub import HfApi

# Hugging Face模型hub用戶名
HF_USERNAME = os.environ.get('HF_USERNAME')
HF_ACCESS_TOKEN = os.environ.get('HF_ACCESS_TOKEN')

# 您想要在模型hub上创建的模型的名稱
model_name = "toxic-comment-classification"
model_description = "model for toxic-comment-classification"
model_local_folder_path = ""

# 建立Hugging Face API
api = HfApi(token=HF_ACCESS_TOKEN)

# 上傳模型到Hugging Face模型hub
api.create_repo(
    repo_id=f'{HF_USERNAME}/{model_name}',
    private=False
)

# 上传模型文件夹到模型hub
api.upload_folder(
    folder_path=model_local_folder_path,
    repo_id=f'{HF_USERNAME}/{model_name}',
    repo_type='model'
)

import gdown

# Google Drive file ID and output path
file_id = '1hGuYthH2MKLZdtWxBfjpQqWNuoBGRIar'
output = 'fine-tuned-abstractive/model.safetensors'

# Download the file
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

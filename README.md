# AI_heath

```bash
conda create -n AI_health python=3.11.11
conda activate AI_health
pip install -r requirements.txt
conda install -c conda-forge libstdcxx-ng
```
add checkpoint to folder AI health

https://drive.google.com/drive/folders/1ebXMkvQCPXTBtGbOunyUhbm3JJSFCPqt

```bash
docker build -t ai-health-server .
docker run -it --rm -p 50051:50051 -e OPENAI_API_KEY="your_key" ai-health-server

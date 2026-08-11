# Local model files

Place the local GGUF model and matching multimodal projection file in this directory. Model weights are ignored by Git.

The Rust desktop runtime communicates with a separately running, OpenAI-compatible llama-server endpoint. The application does not load GGUF files directly and does not require a Python runtime.

Configure the endpoint before launching the desktop app when it differs from the local default:

```powershell
$env:BAODOU_LLAMA_URL = "http://<host>:8765/v1/chat/completions"
```

The expected model is a vision-capable Qwen3.5 2B GGUF with its compatible `mmproj` file. Refer to the model publisher for license and distribution terms.

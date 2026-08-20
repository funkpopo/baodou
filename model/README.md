# Local model files

Place each local GGUF model and its matching multimodal projection file in a
subdirectory of this directory. Model weights are ignored by Git.

The Rust desktop runtime communicates with a separately running, OpenAI-compatible llama-server endpoint. The application does not load GGUF files directly and does not require a Python runtime.

Configure the endpoint before launching the desktop app when it differs from the local default:

```powershell
$env:BAODOU_LLAMA_URL = "http://<host>:8765/v1/chat/completions"
```

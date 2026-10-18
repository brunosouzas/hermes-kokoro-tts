"""Record real G2P output from the pinned Kokoro pipelines into a test fixture (dev tool).

Run with the server's environment, from the repository root:
  HF_HOME=... HF_HUB_OFFLINE=1 <service>/.venv/bin/python -I tests/record_g2p.py tests/fixtures/g2p.json
"""
import json
import sys

from mlx_audio.tts.utils import load_model

TEXTS = {
    "p": [
        "O pipeline terminou e o card do deploy já está no board.",
        "Faz o merge do pull request quando puder.",
        "Revisei os cards e as features.",
        "Abre o (card).",
        "O pull, request ficou para depois.",
        "Manda um e-mail sobre o deploy.",
        "Um copo d'água antes do deploy.",
        "A versão 4.9 do pipeline saiu.",
        "Temos 3 cards no board.",
        "O cardápio chegou.",
        "Ele disse \"card\" duas vezes.",
        "Deploy!",
        "pipeline", "card", "deploy", "board", "merge", "pull", "request", "cards", "features", "e-mail",
        "zzqx",
    ],
    "a": ["pipeline", "card", "cards", "deploy", "board", "merge", "pull request", "pull", "request",
          "features", "e-mail", "zzqx", "Kubernetes", "MuleSoft"],
    "b": ["pipeline", "card", "cards", "deploy", "board", "merge", "pull request", "pull", "request",
          "features", "e-mail", "zzqx", "Kubernetes", "MuleSoft"],
}

model = load_model("mlx-community/Kokoro-82M-bf16", revision="a71e4d38b236d968966a2002c4c895dbd12b1c3c")
out = {}
for lang, texts in TEXTS.items():
    pipeline = model._get_pipeline(lang)
    out[lang] = {}
    for text in texts:
        result = pipeline.g2p(text)
        out[lang][text] = (result[0] if isinstance(result, tuple) else result) or ""
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, sort_keys=True)
print("recorded", sum(len(v) for v in out.values()))

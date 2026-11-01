"""Run this first. Confirms versions, GPU, chat template and that the model answers."""
import sys, torch, transformers
from common import *

print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| transformers", transformers.__version__)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
model, tok = load_model()
print("model:", MODEL_ID, "| layers:", len(model.model.layers),
      "| eos:", tok.eos_token, tok.eos_token_id)
print("--- chat template ---\n" + chat_text(tok, "Hello") + "\n---------------------")
tests = ["What is the capital of France?",
         "भारत की राजधानी क्या है?",
         "Explique brevemente qué es la fotosíntesis."]
for p, o in zip(tests, generate_batch(model, tok, tests, max_new_tokens=40)):
    print("Q:", p, "\nA:", o, "\n")

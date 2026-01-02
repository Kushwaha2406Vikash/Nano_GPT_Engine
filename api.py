from fastapi import FastAPI
from pydantic import BaseModel
import torch
import sentencepiece as spm
from inference_model import model, sp   

app = FastAPI()

class Request(BaseModel):
    text: str
    max_new_tokens: int = 20


@app.post("/generate")
def generate_text(req: Request):
    ids = sp.encode(req.text, out_type=int)
    context = torch.tensor([ids], dtype=torch.long)

    out = model.generate(context, max_new_tokens=req.max_new_tokens)

    input_len = len(ids)
    full_ids = out[0].tolist()

    # ONLY DECODE GENERATED PART
    new_tokens = full_ids[input_len:]

    answer = sp.decode(new_tokens).strip()

    return {
        "question": req.text,
        "answer": answer
    }

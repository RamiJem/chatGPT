from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

import os
import torch


PATH = "app/state_dict_model.pt"

# for s in os.walk("."):
#     print(s)
from app.model import BigramLanguageModel
from app.model import decode
device = "cuda" if torch.cuda.is_available() else "cpu"

model2 = BigramLanguageModel()
model2.load_state_dict(torch.load(PATH))
m2 = model2.to(device)
m2.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m2.generate(context, max_new_tokens=40)[0].tolist())
print(generated_text)

app = FastAPI()



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/")
async def post():
    model2 = BigramLanguageModel()
    model2.load_state_dict(torch.load(PATH))
    m2 = model2.to(device)
    m2.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(m2.generate(context, max_new_tokens=40)[0].tolist())
   # print(generated_text)
    generated_text = generated_text
    return {"message": generated_text}



origins = ["*"]
app = CORSMiddleware(
    app=app,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
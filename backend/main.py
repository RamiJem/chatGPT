from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch


PATH = "state_dict_model.pt"

from model import BigramLanguageModel
device = "cuda" if torch.cuda.is_available() else "cpu"


chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
itos = {i: ch for i, ch in enumerate(chars)}
decode = lambda l: "".join([itos[i] for i in l])

model2 = BigramLanguageModel()
model2.load_state_dict(torch.load(PATH))
m2 = model2.to(device)
m2.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(decode(m2.generate(context, max_new_tokens=10)[0].tolist()))


app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/")
async def post():
    return {"message": "Hello post req"}

import json 
from datasets import load_dataset

def descargar_dataset(limite: int = 2000):
    dataset = load_dataset("allenai/WildChat", split = "train", streaming=True)
    benignos = []
    for item in dataset:
        if not item["toxic"]:
            primer_mensaje = item['conversation'][0]["content"]
            benignos.append(primer_mensaje)
            if len(benignos) >= limite:
                break
            
    with open('data/wildchat_benignos.json', 'w') as f:
        json.dump(benignos,f)
        
        
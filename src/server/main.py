from fastapi import FastAPI
from pydantic import BaseModel
import torch
from peft import PeftModel  
from transformers import  AutoModelForCausalLM, AutoTokenizer


app = FastAPI(title='Servicio de Defensa Adaptativa')


ID_BASE = 'meta-llama/Llama-Guard-3-8B'
tokenizador = AutoTokenizer.from_pretrained(ID_BASE)
modelo_base = AutoModelForCausalLM.from_pretrained(ID_BASE, torch_dtype=torch.bfloat16,device_map ='auto')
modelo_activo = modelo_base


class Consulta(BaseModel):
    prompt: str
    
class Recarga(BaseModel):
    ruta_adaptador: str
    
    
@app.post('/v1/guard')
def evaluar_prompt(consulta: Consulta):
    entradas = tokenizador(f'<human>: {consulta.prompt}\n<bot>:', return_tensors='pt').to('cuda')
    with torch.no_grad():
        salida = modelo_activo.generate(**entradas,max_new_tokens=10)
    veredicto = tokenizador.decode(salida[0], skip_special_tokens=True)
    es_seguro = 'unsafe' not in veredicto.lower()
    return {'prompt': consulta.prompt, 'seguro': es_seguro}


@app.post('/v1/admin/recargar-adaptador')
def recargar_adaptador(datos: Recarga):
    global modelo_activo
    modelo_activo = PeftModel.from_pretrained(modelo_base, datos.ruta_adaptador)
    return {'estado':'adaptador cargado en memoria', 'ruta': datos.ruta_adaptador}

  
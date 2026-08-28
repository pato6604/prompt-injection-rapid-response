import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ServicioGuardia:
    """Servicio de clasificacion de seguridad con soporte para cambio dinamico de adaptadores LoRA"""
    def __init__(self, id_modelo_base: str = "meta-llama/Llama-Guard-3-8B"):
        self.tokenizador = AutoTokenizer.from_pretrained(id_modelo_base)
        self.modelo_base = AutoModelForCausalLM.from_pretrained(
            id_modelo_base,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
    def recargar_adaptador(self, ruta_adaptador:str):
        """Actualiza el adaptador LoRA activo en tiempo real sin reiniciar el modelo base en GPU"""
        self.modelo_activo = PeftModel.from_pretrained(self.modelo_base,ruta_adaptador)
        print(f"Defensa actualizada con el adaptador: {ruta_adaptador}")
        
        
    def es_seguro(self,prompt_usuario:str) -> bool:
        """Clasifica si el prompt recibido es seguro o es un intento de jailbreak"""
        entradas = self.tokenizador(f"<human>:{prompt_usuario}\n<bot>:", return_tensors="pt").to("cuda")
        with torch.no_grad():
            salida = self.modelo_activo.generate(**entradas,max_new_tokens=10)
            
        veredicto = self.tokenizador.decode(salida[0], skip_special_tokens=True)
        
        return "unsafe" not in veredicto.lower()        
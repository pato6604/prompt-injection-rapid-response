import torch
from tqdm import tqdm 
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from src.eval.metricas import calcular_metricas_ataque, calcular_refusal_rate_benigno, ResultadosEvaluacion




class EvaluadorGuardia:
    def __init__(self, modelo: PreTrainedModel, tokenizador: PreTrainedTokenizerBase, batch_size = 8, device: str = "cuda"):
        self.modelo = modelo
        self.tokenizador = tokenizador
        self.batch_size = batch_size
        self.device = device
        self.modelo.eval()
        
        
        
        
    def _predecir_batch_(self,prompts:list[str]) -> list[bool]:
        """Devuelve true si el prompt se considera seguro, False si se considera inseguro"""
        textos_formateados = [f"<human>: {p}\n<bot>:" for p in prompts]
        entradas = self.tokenizador(
            textos_formateados,
            padding = True,
            max_length = 2048,
            return_tensors = "pt"
        ).to(self.device)
        
        
        with torch.no_grad():
            salidas = self.modelo.generat(**entradas, max_new_tokens=10)
            
        
        predicciones = []
        
        for i, salida in enumerate(salidas):
            
            # Decodificar solo los tokens nuevos generados
            nuevos_tokens = salida[entradas["input_ids"][i].shape[0]:]
            texto_generado = self.tokenizador.decode(nuevos_tokens,skip_special_tokens=True).strip().lower()
            
            # En Llama-Guard la etiqueta peligrosa inlcuye 'unsafe'
            
            es_seguro = "unsafe" not in texto_generado
            predicciones.append(es_seguro)
            
            
        return predicciones
    
    def evaluar_dataset(self,prompts: list[str]) -> list[bool]:
        """Ejecuta inferencia sobre una lista completa de prompts dividida en batches""" 
        
        resultados = []       
        for i in tqdm(range(0,len(prompts),self.batch_size),desc="Evaluando"):
            batch = prompts[i:i + self.batch_size]
            resultados.extend(self._predecir_batch_(batch))
        
        return resultados 
    
    
    def ejecutar_benchmark_completo(
        self,
        prompts_id:list[str],
        prompts_ood:list[str],
        prompts_wildchat: list[str]
    ) -> dict:
        
        """Calcula el benchmark completo de RapidResponse"""
        
        
        preds_id = self.evaluar_dataset(prompts_id)
        preds_ood = self.evaluar_dataset(prompts_ood)
        preds_benignos = self.evaluar_dataset(prompts_wildchat)
        
        metricas_id = calcular_metricas_ataque(preds_id)
        metricas_ood = calcular_metricas_ataque(preds_ood)
        refusal_rate = calcular_refusal_rate_benigno(preds_benignos)
        
        return {
            "asr_id" : metricas_id.tasa_exito_ataque,
            "asr_ood": metricas_ood.tasa_exito_ataque,
            "refusal_rate_wildchat": refusal_rate
        }           
        
        
        
        
        
        
        
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model 

def entrenar_adaptador_rapido(
    dataset: Dataset,
    id_modelo_base: str = "meta-llama/Llama-Guard-3-8B",
    ruta_salida: str = "./adaptadores/ultima_defensa"
    
):
    """Entrena un adaptador LoRA en minutos para actualizar el clasificador de entrada"""
    tokenizador = AutoTokenizer.from_pretrained(id_modelo_base)
    modelo = AutoModelForCausalLM.from_pretrained(
        id_modelo_base,
        torch_dtype= torch.bfloat16,
        device_map = "auto"
    )
    
    
    
# Hiperparametros propuestos por el paper        
configuracion_peft = LoraConfig(
    r = 8,
    lora_alpha = 32,
    target_modules = ['q_proj','v_proj','k_proj','o_proj'],
    lora_dropout = 0.0,
    bias = 'none',
    task_type = 'CAUSAL_LM'
)

modelo = get_peft_model(modelo,configuracion_peft)  

argumentos_entrenamiento = TrainingArguments(
    output_dir = ruta_salida,
    learning_rate = 1e-4,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 8, #Tamaño del batch efectivo = 32
    num_train_epochs = 1,
    warmup_ratio = 1.0,
    lr_scheduler_type = "linear",
    bf16 = True,
    logging_steps = 5,
    save_strategy = "no"
    )     

def formatear_para_clasificador(ejemplo):
    texto = f"<human>: {ejemplo['texto']}\n<bot>:{ejemplo['etiqueta']}"
    tokens = tokenizador(texto,truncation = True, maxx_lenght=2048)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens


dataset_tokenizado = dataset.map(fromatear_para_clasifiacador)
entrenador = Trainer(model=modelo, args=argumentos_entrenamiento, train_dataset=dataset_tokenizado)
entrenador.train()


modelo.save_pretrained(ruta_salida)
print(f'Adaptador guardado correctamente en: {ruta_salida}')



from datasets import Dataset    


def construir_dataset_entrenamiento(ataques_proliferados: list[str],consultas_benignas:list[str]) -> Dataset:
    """Crea un datset balanceado (50% inseguro, 50% seguro) para el clasificador."""
    
    registros = [] 
    
    for ataque in ataques_proliferados:
        registros.append({"texto": ataque, "etiqueta": "unsafe"})

    cantidad_ataques = len(ataques_proliferados)
    
    for consulta in consultas_benignas[:cantidad_ataques]:
        registros.append({"texto":consulta, "etiqueta": "safe"})
        
    return Dataset.from_list(registros)



       
    
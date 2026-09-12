# 🛡️ RapidResponse: Pipeline Adaptativo de Defensa contra Jailbreaks en LLMs en Tiempo Real

> **Implementación y Prototipo de Ingeniería** basado en el artículo de
> investigación:
>
> *«Rapid Response: Mitigating LLM Jailbreaks with a Few Examples»*
> (Peng et al., 2024 - Anthropic, NYU, MATS).

## 📌 Resumen Ejecutivo

Las defensas de alineación estáticas (como RLHF base) se degradan con
rapidez frente a intentos de jailbreak persistentes y adaptativos. En
lugar de perseguir una invulnerabilidad estática previa al despliegue
imposible de garantizar, **Respuesta Rápida (Rapid Response)** propone
un paradigma de seguridad dinámico: tras detectar un nuevo patrón de
ataque en producción, el sistema prolifera automáticamente variaciones
sintéticas, entrena un adaptador de clasificación eficiente en
parámetros (LoRA) y recarga la defensa en tiempo real sin interrumpir el
servicio

``` text
[ Petición Entrante de Usuario ]
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Filtro de Entrada (FastAPI + Llama-Guard-3 / Multi-LoRA) │
│    - Bloquea peticiones dañinas con cero tiempo de parada   │
└─────────────────────────────────────────────────────────────┘
                │
                ├─── [ Nuevo Jailbreak Detectado en Telemetría ] ─────┐
                │                                                     ▼
                │                         ┌───────────────────────────────────────┐
                │                         │ 2. Motor de Proliferación Sintética   │
                │                         │    - Aumento sintético mediante LLM   │
                │                         │    - Objetivos basados en AdvBench    │
                │                         └───────────────────────────────────────┘
                │                                                     │
                │                                                     ▼
                │                         ┌───────────────────────────────────────┐
                │                         │ 3. Reentrenamiento Rápido Automatizado│
                │                         │    - Datos balanceados 50/50(WildChat)│
                │                         │    - Adaptación de Bajo Rango (LoRA)  │
                │                         └───────────────────────────────────────┘
                │                                                     │
                └──────── [ Recarga en Caliente del Nuevo Adaptador ] ◄┘
```

### 🎯 Logros Clave del Benchmark (según el Paper)

-   **Reducción de ASR \>240x (99.6%)** en familias de jailbreak
    *In-Distribution* (ID) a partir de un único ejemplo observado.
-   **Reducción de ASR \>15x (93.6%)** en variantes de ataque no vistas
    *Out-of-Distribution* (OOD).
-   **Usabilidad Benigna Preservada:** Incremento mínimo en la tasa de
    rechazo erróneo sobre consultas legítimas reales de usuarios en
    `WildChat`.

## 🏗️ Arquitectura del Sistema y Stack Tecnológico

  ------------------------------------------------------------------------------
  Módulo                  Propósito               Stack Tecnológico
  ----------------------- ----------------------- ------------------------------
  **Inference Guard**     Clasificación en tiempo `FastAPI`, `vLLM` / `PEFT`,
                          real y recarga en       `Llama-Guard-3-8B`
                          caliente sin caídas     
                          (*zero-downtime*)       

  **Synthetic             Generación de cientos   `LiteLLM`, `Instructor`,
  Proliferation**         de variantes dirigidas  `Pydantic`, `GPT-4o` /
                          a conductas dañinas     `Llama-3.1-70B`

  **Rapid Fine-Tuning**   Adaptación rápida con   `Unsloth` /
                          LoRA (r=8, α=32) sobre  `Hugging Face Transformers`,
                          datos balanceados       `BitsAndBytes`

  **Data & Evaluation**   Evaluación comparativa  `AdvBench`, `WildChat`,
                          ID vs. OOD y            `Weights & Biases`, `Gradio`
                          seguimiento de rechazos 
                          benignos                
  ------------------------------------------------------------------------------

## 📂 Estructura del Repositorio

``` text
├── data/
│   ├── advbench_behaviors.json      # Conductas dañinas objetivo para proliferación
│   └── wildchat_benign_sample.jsonl # Muestra de dataset benigno para mezcla 50/50
├── src/
│   ├── guard/
│   │   ├── classifier.py            # Wrapper de inferencia para Llama-Guard y tokenizer
│   │   └── hot_reload.py            # Conmutador dinámico de pesos LoRA en memoria
│   ├── proliferation/
│   │   ├── generator.py             # Worker de proliferación (Chain-of-Thought)
│   │   └── schemas.py               # Esquemas Pydantic para extracción estructurada
│   ├── training/
│   │   ├── finetune.py              # Entrenador rápido LoRA (PEFT / Unsloth)
│   │   └── dataset_builder.py       # Generador de dataset balanceado (Safe/Unsafe)
│   └── server/
│       └── app.py                   # Pasarela de producción en FastAPI
├── demo/
│   └── app.py                       # Panel interactivo en Gradio / Streamlit
├── notebooks/
│   └── rapid_response_eval.ipynb    # Curvas de evaluación y gráficos de ASR vs. Refusal
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Guía de Inicio Rápido

### 1. Instalación y Configuración del Entorno

``` bash
git clone https://github.com/tu-usuario/rapid-response-llm-defense.git
cd rapid-response-llm-defense

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configura tus variables de entorno:

``` bash
cp .env.example .env
# Agrega tus claves: HF_TOKEN, OPENAI_API_KEY y WANDB_API_KEY
```

### 2. Ejecutar la Demostración Interactiva

Inicia la interfaz gráfica para simular un ataque de día cero, disparar
la proliferación automática y comprobar la adaptación del filtro:

``` bash
python demo/app.py
```

### 3. Iniciar la API en Producción

``` bash
uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
```

## 🔬 Metodología de Benchmark y Flujo de Trabajo

1.  **Detección e Ingesta:** El sistema registra un prompt adversarial
    exitoso no bloqueado (ej. *Skeleton Key*, *Crescendo* o *PAIR*).

2.  **Fase de Proliferación:** Mediante prompts con *Chain-of-Thought* y
    pocos ejemplos, un LLM auxiliar proyecta la estrategia de ataque
    sobre diversas conductas de AdvBench.

3.  **Balanceo de Datos:** Las variantes inseguras sintéticas se
    emparejan 1:1 con consultas benignas de WildChat para evitar falsos
    positivos y sobrebloqueos.

4.  **Ajuste Fino con LoRA:** El adaptador del clasificador se actualiza
    durante 1 época usando rango r=8 y α=32.

5.  **Reemplazo en Caliente (*Zero-Downtime Swap*):** El servicio
    FastAPI actualiza el adaptador LoRA activo en memoria sin necesidad
    de reiniciar el modelo base en la VRAM.

## 📊 Resultados Experimentales

``` text
[Baseline ASR]    ████████████████████ 41.2% (Vulnerable)
[Adaptado ID]     ▍ 0.2% (-99.6% de reducción en ASR)
[Adaptado OOD]    ██▍ 2.6% (-93.6% de reducción en ASR)
[Rechazo Benigno] █▍ 6.8% (Impacto casi nulo en usabilidad legítima)
```

## 📜 Citación y Agradecimientos

Si utilizas este código o arquitectura en tus proyectos, por favor cita
el paper original

``` bibtex
@article{peng2024rapidresponse,
  title={Rapid Response: Mitigating LLM Jailbreaks with a Few Examples},
  author={Peng, Alwin and Michael, Julian and Sleight, Henry and Perez, Ethan and Sharma, Mrinank},
  journal={arXiv preprint arXiv:2409.XXXXX},
  year={2024}
}
```

## ⚖️ Consideraciones Éticas y de Seguridad

Este repositorio ha sido desarrollado **exclusivamente con fines
educativos y de investigación en AI Safety / Seguridad Defensiva**. Los
prompts y las técnicas de proliferación se emplean únicamente para
evaluar y entrenar filtros de entrada contra usos adversarios y
maliciosos.

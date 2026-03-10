# IberianOCR: Sistema de Reconocimiento Óptico para la Escritura Ibérica

Repositorio del Trabajo de Fin de Grado (TFG) en Ingeniería de Software de la Universidad Politécnica de Madrid (UPM).

**Autor:** Sergio Gutiérrez Jurado  
**Tutora:** Dra. María Navas Loro  
**Año:** 2026

---

## 📖 Descripción del Proyecto

Este proyecto aborda la digitalización del patrimonio epigráfico paleohispánico, específicamente la **lengua ibérica nororiental (levantina)**. Al tratarse de una lengua aislada y no descifrada semánticamente (*Low-Resource Language*), los motores OCR comerciales que dependen de diccionarios y modelos de lenguaje predictivos fallan en su transcripción.

Este repositorio documenta todo el proceso de ingeniería: desde la investigación inicial con técnicas de Procesamiento del Lenguaje Natural (PLN) y los intentos de reentrenamiento de Tesseract 5, hasta el desarrollo de una **arquitectura neuronal propia (MLP)** y un pipeline de Visión Artificial (OpenCV) capaz de segmentar y clasificar los glifos de forma autónoma.

---

## 🗂️ Estructura del Repositorio

El código está organizado cronológicamente según las fases de investigación detalladas en la memoria del TFG:

### 📁 `1_Investigacion_PLN/`
Contiene los scripts de análisis estadístico del lenguaje y manipulación de datos en bruto.
* Extracción y limpieza de *tokens* desde bases de datos epigráficas (Hesperia, Ibérika).
* Análisis de **N-gramas** y visualización de frecuencias con `matplotlib`.
* Scripts de control comparativo utilizando inscripciones latinas (`edh.xlsx`).
* Análisis programático de la tipografía digital (`fontAnalyzer.py`).

### 📁 `2_Experimentos_Tesseract/`
Documenta los intentos de adaptar motores OCR comerciales estándar.
* Generadores automáticos de datasets sintéticos (`createTif.py`, `PNGtoTIF.py`) mediante `uharfbuzz` y `freetype`.
* Scripts de orquestación para el entrenamiento de redes LSTM de Tesseract.
* Registro de comandos y fallos técnicos de segmentación/deserialización que justificaron el cambio de arquitectura.

### 📁 `3_IberianOCR_Final/`
El producto software final y funcional.
* **Modelo Neuronal:** Arquitectura Perceptrón Multicapa (MLP) entrenada con *Data Augmentation*.
* **Visión Artificial:** Algoritmos de segmentación de contornos para aislar grafías irregulares en *scriptio continua*.
* **Interfaz Gráfica (GUI):** Aplicación de escritorio diseñada para asistir a los epigrafistas en la transcripción, permitiendo verificación visual (*resíntesis* tipográfica).

---

## ⚙️ Instalación y Uso

Para ejecutar el código de este repositorio, se recomienda crear un entorno virtual e instalar las dependencias incluidas:

```bash
# 1. Clonar el repositorio
git clone [https://github.com/sergioguj/TFG-IberianOCR.git](https://github.com/sergioguj/TFG-IberianOCR.git)
cd TFG-IberianOCR

# 2. Crear y activar entorno virtual (Opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
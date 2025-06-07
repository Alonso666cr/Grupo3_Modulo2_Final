# Proyecto de IA - grupo3-modulo2-final

## Descripción
Proyecto de Machine Learning y Procesamiento de Lenguaje Natural desarrollado en Python. El notebook principal implementa análisis de datos, modelos de clasificación con scikit-learn, y procesamiento de texto usando NLTK, Gensim y LangChain.

## Archivo Principal
- **`grupo3-modulo2-final.ipynb`** - Notebook principal del proyecto

## Requisitos del Sistema
- Python 3.8 o superior
- Anaconda o Miniconda
- pip (gestor de paquetes de Python)
- Git

## Instalación y Configuración

### 📋 Opción 1: Instalación Automática (Recomendada)
```bash
# 1. Clona el repositorio
git clone [URL-de-tu-repositorio]
cd [nombre-del-repositorio]

# 2. Abre Anaconda Prompt (NO CMD de Windows)

# 3. Navega al proyecto
cd "ruta/completa/al/proyecto"

# 4. Configuración completa automática
setup_anaconda.bat
```

### ⚙️ Opción 2: Usando Make (Si tienes make instalado)
```bash
# En Anaconda Prompt
make setup
```

### 🔧 Opción 3: Instalación Manual
```bash
# En Anaconda Prompt
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
python check_versions.py
```

## Uso del Proyecto

### 🚀 Ejecutar el Notebook
```bash
# Opción 1: Abrir el notebook específico
jupyter notebook grupo3-modulo2-final.ipynb

# Opción 2: Abrir Jupyter y navegar al archivo
jupyter notebook

# Opción 3: Usar make
make notebook
```

### 📊 Verificar Instalación
```bash
# Verificar que todas las librerías funcionan
python check_versions.py

# O usar make
make check
```

## Comandos Disponibles

### Con Make (desde Anaconda Prompt):
- `make help` - Muestra todos los comandos disponibles
- `make setup` - Configuración completa del proyecto  
- `make check` - Verificar versiones instaladas
- `make test` - Probar todas las importaciones
- `make notebook` - Iniciar Jupyter con el notebook
- `make clean` - Limpiar archivos temporales
- `make update_dependencies` - Actualizar requirements.txt

### Con Scripts Batch:
- `setup_anaconda.bat` - Configuración completa automática
- `python check_versions.py` - Verificar instalación

## Estructura del Proyecto
```
├── grupo3-modulo2-final.ipynb    # Notebook principal
├── requirements.in               # Dependencias principales  
├── requirements.txt              # Dependencias completas (generado)
├── setup_anaconda.bat           # Script de configuración
├── check_versions.py            # Script de verificación
├── Makefile                     # Comandos automatizados
├── .gitignore                   # Archivos ignorados por Git
├── CODEOWNERS                   # Propietarios del código
└── README.md                    # Este archivo
```

## Librerías Principales Utilizadas

| Librería | Versión | Propósito |
|----------|---------|-----------|
| pandas | 2.2.2 | Análisis de datos |
| numpy | 1.26.4 | Cálculos numéricos |
| scikit-learn | 1.5.1 | Machine Learning |
| matplotlib | 3.9.2 | Visualización |
| seaborn | 0.13.2 | Visualización estadística |
| nltk | 3.9.1 | Procesamiento de lenguaje natural |
| gensim | 4.3.3 | Modelado de temas |
| langchain | 0.3.25 | Procesamiento de texto avanzado |

## Ejecutar en Diferentes Plataformas

### 🔵 Google Colab
1. Sube el notebook `grupo3-modulo2-final.ipynb` a Google Colab
2. Sube también el archivo `requirements.txt`
3. En la primera celda ejecuta:
```python
!pip install -r requirements.txt
```
4. Ejecuta el notebook normalmente

### 🟠 Kaggle
1. Sube el notebook a Kaggle
2. La mayoría de dependencias ya están instaladas
3. Para dependencias faltantes:
```python
!pip install langchain gensim
```
4. Ejecuta el notebook

### 🟢 Jupyter Local
Sigue las instrucciones de instalación y ejecuta `jupyter notebook`

### 🟡 Anaconda Navigator
1. Abre Anaconda Navigator
2. Instala las dependencias usando Anaconda Prompt
3. Abre Jupyter desde Navigator
4. Navega al notebook

## Verificación de la Instalación

### ✅ Verificación Rápida
```bash
python check_versions.py
```

### ✅ Verificación Completa
```bash
make test
```

### ✅ Verificación Manual
```python
# Ejecuta en Python/Jupyter
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import nltk, gensim
print("✅ Todas las importaciones funcionan")
```

## Troubleshooting

### 🐛 Problemas Comunes

**Error: "make no se reconoce"**
- Solución: Usa `setup_anaconda.bat` en lugar de make

**Error: "conda no disponible"**
- Solución: Ejecuta desde Anaconda Prompt, no desde CMD

**Error: "ModuleNotFoundError"**
- Solución: Ejecuta `setup_anaconda.bat` o `pip install -r requirements.txt`

**Error con NLTK data**
- Solución: Ejecuta `make nltk-setup` o el script manual:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')
```

### 🔄 Actualizar Dependencias
```bash
# Si necesitas agregar nuevas librerías:
# 1. Agrégalas a requirements.in
# 2. Ejecuta:
make update_dependencies
make install
```

## Reproducibilidad
Este proyecto está configurado para ser **100% reproducible**:
- ✅ Versiones específicas de todas las dependencias
- ✅ Scripts de configuración automática  
- ✅ Verificación de instalación
- ✅ Documentación completa
- ✅ Compatible con múltiples plataformas

## Contribuir

### 🔒 Proceso de Contribución (Con Restricciones de @alonso666cr)
1. Fork el repositorio
2. Crea una rama para tu feature: `git checkout -b feature/nueva-funcionalidad`
3. Realiza tus cambios y haz commit
4. Push de tu rama: `git push origin feature/nueva-funcionalidad`
5. Crea un Pull Request en GitHub
6. **@alonso666cr debe aprobar** antes del merge (obligatorio)
7. Solo después de aprobación se puede hacer merge

### ⚙️ Configurar Restricciones de Pull Request
**IMPORTANTE:** Para activar restricciones automáticas, @alonso666cr debe:
1. Ir a **Settings** → **Branches** en GitHub
2. Añadir regla para rama `main`
3. Marcar **"Require review from CODEOWNERS"**
4. Esto **bloquea** todos los merges sin aprobación

**Estado actual:** 
- ✅ CODEOWNERS configurado para @alonso666cr
- ⚙️ Branch Protection Rules: **Configurar manualmente en GitHub**

**Sin Branch Protection:** CODEOWNERS solo sugiere @alonso666cr como revisor  
**Con Branch Protection:** **Bloquea** automáticamente merges sin aprobación de @alonso666cr

## Notas Importantes
- **Siempre usar Anaconda Prompt** para comandos de instalación
- El proyecto está optimizado para el entorno Anaconda
- Todas las versiones están probadas y funcionando
- El notebook incluye procesamiento de texto avanzado con LangChain

## Soporte
Si encuentras problemas:
1. Ejecuta `python check_versions.py` para diagnosticar
2. Ejecuta `setup_anaconda.bat` para reconfigurar
3. Revisa la sección Troubleshooting arriba
4. Verifica que estés usando Anaconda Prompt

## Contacto
[Tu información de contacto]

## Licencia
[Especifica tu licencia aquí]
# Makefile para proyecto grupo3-modulo2-final
# Diseñado para trabajar con Anaconda
# IMPORTANTE: Ejecutar comandos desde Anaconda Prompt

.PHONY: help install update_dependencies clean setup test notebook check

# Comando por defecto
help:
	@echo "Comandos disponibles para grupo3-modulo2-final:"
	@echo "  make setup            - Configuración completa del proyecto"
	@echo "  make check            - Verificar versiones instaladas"
	@echo "  make update_dependencies - Actualizar requirements.txt"
	@echo "  make install          - Instalar dependencias"
	@echo "  make test             - Probar todas las importaciones"
	@echo "  make notebook         - Iniciar Jupyter con el notebook"
	@echo "  make clean            - Limpiar archivos temporales"
	@echo "  make help             - Mostrar esta ayuda"
	@echo ""
	@echo "NOTA: Ejecutar desde Anaconda Prompt, no desde CMD"

# Verificar versiones actuales
check:
	@echo "Verificando versiones en tu entorno de Anaconda..."
	python check_versions.py

# Actualizar requirements.txt desde requirements.in
update_dependencies:
	@echo "Actualizando requirements.txt..."
	pip install pip-tools
	pip-compile requirements.in --no-strip-extras
	@echo "requirements.txt actualizado exitosamente"

# Instalar dependencias
install:
	@echo "Instalando dependencias..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencias instaladas exitosamente"

# Probar todas las importaciones del notebook
test:
	@echo "Probando importaciones del notebook..."
	python -c "from datetime import datetime; import pandas as pd; import numpy as np; import re; from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score; from sklearn.linear_model import LogisticRegression; from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve; from sklearn.preprocessing import LabelEncoder; import matplotlib.pyplot as plt; import seaborn as sns; from langchain.text_splitter import SentenceTransformersTokenTextSplitter; import nltk; from nltk.corpus import stopwords; from nltk.tokenize import word_tokenize; from nltk.stem import WordNetLemmatizer; import gensim; from gensim import corpora; from gensim.models import LdaModel; import warnings; print('✓ Todas las importaciones del notebook funcionan correctamente')"

# Configuración completa
setup: update_dependencies install check test
	@echo "Configuración completa terminada"
	@echo "Proyecto listo para usar"

# Iniciar Jupyter con el notebook específico
notebook:
	@echo "Iniciando Jupyter Notebook..."
	jupyter notebook grupo3-modulo2-final.ipynb

# Limpiar archivos temporales
clean:
	@echo "Limpiando archivos temporales..."
	-find . -name "*.pyc" -delete 2>/dev/null
	-find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
	-find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null
	@echo "Limpieza completada"

# Crear environment.yml para conda (alternativa)
conda-export:
	@echo "Exportando entorno conda..."
	conda env export > environment.yml
	@echo "environment.yml creado"

# Verificar que NLTK data está disponible
nltk-setup:
	@echo "Configurando NLTK data..."
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab'); print('NLTK data configurado')"
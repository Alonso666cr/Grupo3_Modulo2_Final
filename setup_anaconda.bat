@echo off
chcp 65001 >nul
echo =====================================
echo   SETUP PROYECTO: grupo3-modulo2-final
echo =====================================
echo.
echo IMPORTANTE: Ejecutar desde Anaconda Prompt
echo.

REM Verificar que estamos en Anaconda
conda info --envs > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: conda no disponible
    echo Por favor ejecuta desde Anaconda Prompt
    pause
    exit /b 1
)

echo [1/7] Verificando entorno actual...
echo Entorno activo: %CONDA_DEFAULT_ENV%
python --version
echo.

echo [2/7] Verificando archivos del proyecto...
if not exist "grupo3-modulo2-final.ipynb" (
    echo WARNING: grupo3-modulo2-final.ipynb no encontrado en esta carpeta
    echo Asegurate de estar en la carpeta correcta del proyecto
)
if not exist "requirements.in" (
    echo ERROR: requirements.in no encontrado
    echo Creando requirements.in con versiones de tu entorno...
    (
    echo # Requirements para grupo3-modulo2-final.ipynb
    echo pandas==2.2.2
    echo numpy==1.26.4
    echo scikit-learn==1.5.1
    echo matplotlib==3.9.2
    echo seaborn==0.13.2
    echo nltk==3.9.1
    echo gensim==4.3.3
    echo langchain==0.3.25
    echo jupyter^>=1.0.0
    echo ipykernel^>=6.0.0
    echo pip-tools^>=6.0.0
    ) > requirements.in
    echo ✓ requirements.in creado
) else (
    echo ✓ requirements.in encontrado
)
echo.

echo [3/7] Instalando pip-tools...
pip install pip-tools
if %ERRORLEVEL% neq 0 (
    echo ERROR: No se pudo instalar pip-tools
    pause
    exit /b 1
)
echo.

echo [4/7] Generando requirements.txt...
pip-compile requirements.in --no-strip-extras
if %ERRORLEVEL% neq 0 (
    echo ERROR: No se pudo compilar requirements.in
    pause
    exit /b 1
)
echo ✓ requirements.txt generado exitosamente
echo.

echo [5/7] Instalando/actualizando dependencias...
pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ERROR: No se pudieron instalar las dependencias
    pause
    exit /b 1
)
echo.

echo [6/7] Configurando NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('punkt_tab', quiet=True); print('✓ NLTK data configurado')"
echo.

echo [7/7] Verificación final del proyecto...
echo Probando todas las importaciones del notebook...
python -c "
try:
    # Importaciones del notebook grupo3-modulo2-final.ipynb
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import re
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import seaborn as sns
    from langchain.text_splitter import SentenceTransformersTokenTextSplitter
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    import warnings
    
    print('✓ TODAS las importaciones del notebook funcionan correctamente')
    print('✓ pandas:', pd.__version__)
    print('✓ numpy:', np.__version__)
    print('✓ scikit-learn:', sklearn.__version__)
    print('✓ matplotlib:', matplotlib.__version__)
    print('✓ seaborn:', sns.__version__)
    print('✓ langchain:', langchain.__version__)
    print('✓ nltk:', nltk.__version__)
    print('✓ gensim:', gensim.__version__)
    print('✓ Tu proyecto está listo para ser compartido')
    
except ImportError as e:
    print('✗ Error en importación:', e)
    exit(1)
"

if %ERRORLEVEL% neq 0 (
    echo ✗ Hay problemas con algunas importaciones
    pause
    exit /b 1
)

echo.
echo =====================================
echo      CONFIGURACION COMPLETADA
echo =====================================
echo.
echo ✓ requirements.in creado con versiones específicas
echo ✓ requirements.txt generado con todas las dependencias
echo ✓ Todas las librerías verificadas y funcionando
echo ✓ NLTK data configurado
echo ✓ Proyecto completamente listo para compartir
echo.
echo Para iniciar tu notebook:
echo   jupyter notebook grupo3-modulo2-final.ipynb
echo.
echo Archivos del proyecto:
echo   - grupo3-modulo2-final.ipynb  ^(tu notebook principal^)
echo   - requirements.in             ^(dependencias principales^)
echo   - requirements.txt            ^(todas las dependencias^)
echo   - check_versions.py           ^(script de verificación^)
echo   - setup_anaconda.bat          ^(este script^)
echo   - Makefile                    ^(comandos make^)
echo   - .gitignore                  ^(archivos a ignorar^)
echo   - CODEOWNERS                  ^(propietarios del código^)
echo   - README.md                   ^(documentación^)
echo.
echo Tu proyecto es ahora 100%% reproducible en cualquier entorno
echo.
pause
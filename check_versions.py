#!/usr/bin/env python3
"""
Script para verificar versiones de librerías para grupo3-modulo2-final.ipynb
Verifica que todas las importaciones del notebook funcionen correctamente
"""

def check_library_versions():
    """Verifica las versiones de todas las librerías necesarias"""
    print('=' * 60)
    print('   VERIFICACIÓN DE LIBRERÍAS - grupo3-modulo2-final')
    print('=' * 60)
    print()

    # Lista de librerías principales con sus alias
    libraries = [
        ('pandas', 'pd'),
        ('numpy', 'np'), 
        ('sklearn', None),
        ('matplotlib', None),
        ('seaborn', 'sns'),
        ('langchain', None),
        ('nltk', None),
        ('gensim', None)
    ]

    print('📦 VERSIONES DE LIBRERÍAS PRINCIPALES:')
    print('-' * 40)
    
    missing_libraries = []
    
    for lib_name, alias in libraries:
        try:
            if alias:
                exec(f"import {lib_name} as {alias}")
                version = eval(f"{alias}.__version__")
            else:
                exec(f"import {lib_name}")
                version = eval(f"{lib_name}.__version__")
            
            print(f'✓ {lib_name:<12} == {version}')
        except ImportError:
            print(f'✗ {lib_name:<12} : NO INSTALADO')
            missing_libraries.append(lib_name)
        except AttributeError:
            print(f'✓ {lib_name:<12} : INSTALADO (sin versión disponible)')

    print()
    
    # Verificar importaciones específicas del notebook
    print('🔍 VERIFICANDO IMPORTACIONES ESPECÍFICAS DEL NOTEBOOK:')
    print('-' * 50)
    
    import_tests = [
        {
            'name': 'scikit-learn submodules',
            'imports': [
                'from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score',
                'from sklearn.linear_model import LogisticRegression', 
                'from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve',
                'from sklearn.preprocessing import LabelEncoder'
            ]
        },
        {
            'name': 'langchain text splitter',
            'imports': ['from langchain.text_splitter import SentenceTransformersTokenTextSplitter']
        },
        {
            'name': 'nltk modules',
            'imports': [
                'from nltk.corpus import stopwords',
                'from nltk.tokenize import word_tokenize', 
                'from nltk.stem import WordNetLemmatizer'
            ]
        },
        {
            'name': 'gensim modules', 
            'imports': [
                'from gensim import corpora',
                'from gensim.models import LdaModel'
            ]
        },
        {
            'name': 'matplotlib/seaborn',
            'imports': [
                'import matplotlib.pyplot as plt',
                'import seaborn as sns'
            ]
        }
    ]
    
    failed_imports = []
    
    for test in import_tests:
        try:
            for import_statement in test['imports']:
                exec(import_statement)
            print(f"✓ {test['name']}")
        except ImportError as e:
            print(f"✗ {test['name']}: {e}")
            failed_imports.append(test['name'])
    
    print()
    
    # Verificar imports estándar de Python
    print('🐍 VERIFICANDO MÓDULOS ESTÁNDAR DE PYTHON:')
    print('-' * 40)
    
    standard_modules = ['datetime', 're', 'warnings']
    for module in standard_modules:
        try:
            exec(f"import {module}")
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}: NO DISPONIBLE")
    
    print()
    print('=' * 60)
    
    # Resumen final
    if not missing_libraries and not failed_imports:
        print('🎉 ¡EXCELENTE! Todas las librerías están instaladas y funcionando')
        print('   Tu notebook grupo3-modulo2-final.ipynb debería ejecutarse sin problemas')
    else:
        print('⚠️  ATENCIÓN: Faltan algunas dependencias')
        if missing_libraries:
            print(f'   Librerías faltantes: {", ".join(missing_libraries)}')
        if failed_imports:
            print(f'   Importaciones fallidas: {", ".join(failed_imports)}')
        print('   Ejecuta: pip install -r requirements.txt')
    
    print('=' * 60)

def test_notebook_imports():
    """Prueba todas las importaciones del notebook de una vez"""
    print()
    print('🧪 PRUEBA COMPLETA DE IMPORTACIONES DEL NOTEBOOK:')
    print('-' * 50)
    
    try:
        # Todas las importaciones del notebook grupo3-modulo2-final.ipynb
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
        
        print('✅ TODAS las importaciones del notebook funcionan perfectamente')
        print('✅ Tu proyecto está 100% listo para ser ejecutado por cualquier persona')
        return True
        
    except ImportError as e:
        print(f'❌ Error en importación: {e}')
        print('   Ejecuta setup_anaconda.bat para solucionar')
        return False

if __name__ == "__main__":
    check_library_versions()
    success = test_notebook_imports()
    
    print()
    if success:
        print('🚀 ¡Tu entorno está perfectamente configurado!')
    else:
        print('🔧 Tu entorno necesita configuración adicional')
        
    print()
    print('Para obtener ayuda adicional, ejecuta:')
    print('  setup_anaconda.bat  (configuración completa)')
    print('  make help          (ver todos los comandos disponibles)')
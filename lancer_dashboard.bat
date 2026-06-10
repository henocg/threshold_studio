@echo off
chcp 65001 >nul
title Threshold Studio — LSN Ré Walbaum

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║   Threshold Studio  —  LSN Re Walbaum       ║
echo  ║   Detection du seuil de sinistre grave       ║
echo  ╚══════════════════════════════════════════════╝
echo.

REM Chemin vers Python Anaconda (utilisateur)
set PYTHON=%USERPROFILE%\AppData\Local\anaconda3\python.exe
set STREAMLIT=%USERPROFILE%\AppData\Local\anaconda3\Scripts\streamlit.exe

REM Aller dans le dossier du dashboard
cd /d "%~dp0"

REM Vérifier que Python existe
if not exist "%PYTHON%" (
    echo [ERREUR] Python Anaconda non trouve dans :
    echo   %PYTHON%
    echo.
    echo Essai avec conda systeme...
    set PYTHON=C:\ProgramData\Anaconda3\python.exe
    set STREAMLIT=C:\ProgramData\Anaconda3\Scripts\streamlit.exe
)

REM Vérifier streamlit
if not exist "%STREAMLIT%" (
    echo [INFO] Streamlit non trouve, installation en cours...
    "%PYTHON%" -m pip install streamlit plotly pandas scipy openpyxl --quiet
    echo Installation terminee.
    echo.
)

echo  Lancement du dashboard sur http://localhost:8501
echo  Appuyez sur Ctrl+C pour arreter.
echo.
"%PYTHON%" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false

pause

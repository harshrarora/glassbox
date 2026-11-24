@echo off
pushd "%~dp0"
python -m streamlit run glassbox/app.py
popd
pause

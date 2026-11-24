Push-Location -LiteralPath $PSScriptRoot
try {
    python -m streamlit run glassbox/app.py
}
finally {
    Pop-Location
}

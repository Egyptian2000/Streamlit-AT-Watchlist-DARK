
# Portfolio Sizer — Grey Background / Dark Red Text (Fixed CSS)

This build removes Python f-string interpolation in the CSS to avoid Streamlit Cloud NameError issues.

## Run
- `pip install -r requirements.txt`
- `streamlit run app.py`

## Deploy (Streamlit Cloud)
- Push these files to a GitHub repo (repo root should contain `app.py`, `requirements.txt`, `.streamlit/config.toml`)
- On Streamlit Cloud: New app → select repo → main file `app.py` → Deploy

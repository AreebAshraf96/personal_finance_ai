# Personal Finance Web (Phase 1)
Flask + MongoDB + Auth baseline to evolve the Streamlit prototype into a real website.

## Quickstart
1. Create a virtualenv (recommended) and activate it.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill `MONGO_URI` and `SECRET_KEY`.
4. `python app.py`
5. Open http://127.0.0.1:5000

## Notes
- Do NOT commit `.env` to Git.
- If MongoDB Atlas blocks your IP, add your public IP in the Atlas Network Access panel.

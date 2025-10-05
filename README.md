This is a packaged subset of the project containing the Streamlit app and minimal artifacts to run the web app. Copy the original processed dataset into `final/data/processed/` before running.

To run:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Comprehensive_NASA_Space_Apps_App.py

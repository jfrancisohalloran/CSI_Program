# CSI_Forecast
Child Saving Institute Forecast Program (John O'Halloran, Sai Lavanya Puthila, and Kelle Springer)

Clone the repo
git clone https://github.com/jfrancisohalloran/CSI_Forecast.git
cd CSI_Forecast

Verify you have Python 3.10
python3.10 --version   # or python --version if that's mapped to 3.10; if not, then we need that to be downloaded and used for this program due to version mismatch. 

Create & activate a virtual environment
On Linux / macOS:
python3.10 -m venv .venv
source .venv/bin/activate

On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt

Run the pipeline
--start-date YYYY-MM-DD   Choose your forecast start date.
If you omit it, you’ll get a GUI picker (or default to today).
--force-refresh           Clears the cached “Grouped_Staff_Requirements.xlsx”
and re-parses all raw sign‑in Excel files.
python -m attendance_pipeline.main --start-date 2025-05-12 --force-refresh

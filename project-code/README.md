# ResStock Energy Planner — Web App

Flask web app that loads NREL ResStock building-stock data from the public OEDI S3 bucket,
trains heating and cooling load models, and lets users predict annual energy use from
building characteristics.

## Setup

```bash
cd energy_app
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## First-run time

On first launch the app fetches a state-level ResStock parquet file (~50–200 MB) from S3
and trains two Random Forest models. Expect **60–120 seconds** before the UI becomes
interactive. Subsequent runs of the same state are instant (the training is in-memory;
add a pickle cache to `app.py` if you want persistence across restarts).

## File structure

```
energy_app/
├── app.py                  # Flask backend — data loading, model training, API routes
├── requirements.txt
├── templates/
│   └── index.html          # Single-page UI
└── static/
    ├── css/style.css
    └── js/app.js           # Polling, form handling, Chart.js rendering
```

## API endpoints

| Method | Path           | Description                                  |
|--------|----------------|----------------------------------------------|
| GET    | `/`            | Serve the UI                                 |
| GET    | `/api/status`  | `{ready, loading, error, state}`             |
| POST   | `/api/load`    | `{state}` — trigger data load for a state   |
| GET    | `/api/choices` | Dropdown options from loaded data            |
| POST   | `/api/predict` | `{fuel, foundation, floor_area, setpoint, stories}` → predictions |

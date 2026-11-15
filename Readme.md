# QuietCount

QuietCount estimates whether a room contains zero, one, two or three people from environmental sensor readings. A facilities team could use a validated estimator to understand room utilisation without installing cameras. This project asks how much predictive performance is retained when temperature, sound and motion sensors are omitted. 

## Data

The bundled original CSV is the [UCI Room Occupancy Estimation dataset](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation). It contains 10,129 readings from one controlled room, with manually recorded occupancy counts. The experiments used no HVAC. The CSV has seven calendar dates, including dates containing only empty-room readings. There were no missing cells, duplicate rows or duplicate timestamps in this snapshot. Attribution and the exact file checksum are in the source and data notes.

## Results

| Sensor inputs | Best validation model | Validation macro F1 |
|---|---|---:|
| Majority-class baseline | Always predicts zero | 0.234 |
| One light channel and CO₂ | Random forest, depth 8 | 0.782 |
| Four light channels and CO₂ | Logistic regression | 0.999 |
| Full set, including sound, temperature, CO₂ slope and motion | Logistic regression | 0.990 |

Training uses December 22, 2017. Validation uses December 23–26, 2017. After selecting the smallest feature set within 0.03 macro F1 of the best validation candidate, the training script refits that pipeline on the combined December rows. January 10–11, 2018 are held out. Date and time define the split and are excluded from prediction.

On the holdout, the selected model achieved **0.475 macro F1 vs 0.308 for the baseline** over the three classes present (0, 2 and 3). Accuracy was **87.4% vs 85.6%**. The model classified 184 of 201 three-person readings as empty, and only 16 correctly as three. Empty readings constitute 1,751 of 2,045 holdout rows, which explains why accuracy appears strong despite those misses. The holdout has no single-person readings. For transparency, the fixed-four-label macro F1 with zero for the absent class is 0.357. See the full candidate results, label support and confusion matrices in `artifacts/metrics.json`.


## Reproduce training

Requires Python 3.12. Run commands from the repository root. On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python data_source.py
python train.py
```

The CSV is included for offline access. If it is removed, `data_source.py` downloads the original from UCI and verifies its SHA-256 checksum. The notebook `analysis.ipynb` presents data checks, EDA and results from the training script. To rerun the notebook or run the tests, install `requirements-dev.txt` first.

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
jupyter lab analysis.ipynb
```

## Run the model service

The included model was trained by `train.py`. Start the API locally or run the Docker commands below.

```sh
python -m uvicorn predict:app --host 127.0.0.1 --port 8000
```

```sh
docker build -t quietcount .
docker run --rm -p 127.0.0.1:8000:8000 quietcount
```

In another terminal:

```sh
curl -fsS http://127.0.0.1:8000/health
curl -fsS -H 'Content-Type: application/json' \
  --data @examples/request.json http://127.0.0.1:8000/predict
```

Expected prediction: `{"occupancy_count":0,"model":"ambient/logistic/-"}`. The input uses four light measurements (lux) and one CO₂ measurement (ppm), corresponding to the original sensor positions. Swagger API documentation is available at [localhost:8000/docs](http://127.0.0.1:8000/docs). The API was verified over local HTTP. Docker build and cloud hosting require separate verification.


### Optional cloud deployment

The included `render.yaml` describes a Docker web service and its health endpoint. In Render, create a Blueprint from the public repository and review the offered plan and billing before deploying. Once deployed, call `/health` and `/predict` at its HTTPS URL using the same example request. Record the live URL and actual response in the README.


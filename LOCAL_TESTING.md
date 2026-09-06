# Local testing on this Mac

The project uses Python 3.12 in `venv`, with the pinned SEN2SR Git submodule installed in editable mode. On Apple Silicon, the web app automatically runs SEN2SR-Lite on the Apple GPU through PyTorch MPS. Copernicus credentials are configured in the ignored local `.env` with owner-only permissions; the active catalog is Copernicus CDSE.

## Start the web app

```bash
cd /Users/pranshubansal/Documents/ChatGPT/llg
source venv/bin/activate
python scripts/run_web.py --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and click **Load Sample Scene** to run the bundled Mountain Lake image through SEN2SR-Lite. Results include comparison overlays and GeoTIFF/PNG downloads. Stop the server with Ctrl+C in its terminal.

## Run tests without Copernicus credentials

```bash
venv/bin/python -m pytest -q -k 'not test_cdse_provider_authentication_and_search'
```

This command runs 113 tests, covering real CPU and MPS inference, GeoTIFF metadata preservation, scientific analysis, and the web job/preview/download workflow. The excluded test performs live Copernicus OAuth and catalog requests.

To run all 114 tests with the configured local credentials:

```bash
venv/bin/python - <<'PY'
from dotenv import load_dotenv
import pytest
load_dotenv('.env', override=True)
raise SystemExit(pytest.main(['-q', '--tb=no']))
PY
```

SEN2SR-Lite is the recommended model for routine interactive testing. Pass
`--device cpu` only when you explicitly need the CPU path.
On this Mac, the 256×256 sample completed model inference in about 1.18 seconds
on MPS versus 5.57 seconds on CPU, with an output RMSE of `1.58e-7` between backends.

## Setup notes

Both model variants are downloaded under `checkpoints/`. The upstream checkpoint loader imports `matplotlib`, which was missing from the original dependency lists; it has been added to `requirements.txt` and `pyproject.toml`.

To recreate the environment using the installed `uv`:

```bash
uv venv --python 3.12 --seed venv
git submodule update --init --recursive
uv pip install --python venv/bin/python -r requirements.txt -e . 'setuptools<81'
uv pip install --python venv/bin/python -e third_party/SEN2SR --no-deps
venv/bin/python scripts/download_checkpoints.py --model all
```

When creating `.env` from `.env.example`, replace placeholder credentials with empty values to use AWS Earth Search, or enter your own valid Copernicus credentials. Local sample testing needs neither account credentials nor a live satellite catalog.

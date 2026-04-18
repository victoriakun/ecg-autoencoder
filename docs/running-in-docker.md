# Running the ECG real-time pipeline in Docker

## Headless mode (any platform)

```bash
docker compose build
docker compose up
# Logs and SQLite event database persist in ./logs
```

## GUI mode

**Linux:**
```bash
xhost +local:docker
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ecg-rt python realtime_app.py
```

**macOS / Windows:** GUI-in-Docker is not officially supported in this
project. Run the GUI natively and use Docker only for headless evaluation
and CI. This is called out in the thesis as a known limitation.

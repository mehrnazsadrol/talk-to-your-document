set -e

uvicorn app.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

for i in $(seq 1 30); do
  curl -fsS http://127.0.0.1:8000/health && break
  sleep 1
done

streamlit run frontend/app.py \
  --server.address=0.0.0.0 \
  --server.port=7860 \
  --server.headless=true \
  --browser.gatherUsageStats=false &
UI_PID=$!

trap "kill $API_PID $UI_PID" INT TERM
wait -n $API_PID $UI_PID

source .venv/bin/activate

export PGPASSWORD=dbos
python3 start_postgres_docker.py

export $(cat .env | xargs)
dbos migrate
dbos start

ngrok http http://localhost:8000
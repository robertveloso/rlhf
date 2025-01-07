from fastapi import FastAPI
from dbos import DBOS, load_config
from imdbbot.app.slack_bot import app as slack_app

app = FastAPI()
dbos = DBOS(fastapi=app)

dbos_config = load_config()

app.include_router(slack_app.router)

@app.get("/")
def root():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    DBOS.launch()
from fastapi import FastAPI
from dbos import DBOS, load_config
from app.slack_bot import app as slack_app

# Initialize FastAPI and DBOS first
app = FastAPI()
dbos = DBOS(fastapi=app)

dbos_config = load_config()

# # Initialize sentiment model after DBOS
# try:
#     sentiment_model = SentimentModel()
# except OSError:
#     print("Custom model not found, skipping...")

# Mount the Slack bot
app.include_router(slack_app.router)

@app.get("/")
def root():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    # import uvicorn
    DBOS.launch()
    # uvicorn.run(app, host="0.0.0.0", port=8000)
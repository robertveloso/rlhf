from slack_bolt import App
from fastapi import Body, FastAPI
from fastapi import Request as FastAPIRequest

from typing import Dict, Any
from dbos import DBOS, Queue, SetWorkflowID
import os
import datetime

from app.movie_db import MovieDatabase
from app.sentiment import SentimentModel
import uuid

app = FastAPI()

# Initialize Slack app
slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    logger=DBOS.logger
)

# Queue for processing feedback
feedback_queue = Queue("movie_feedback_queue", limiter={"limit": 300, "period": 60}, concurrency=1)
movie_recommendation_queue = Queue("movie_recommendation_queue", limiter={"limit": 300, "period": 60}, concurrency=1)
model_training_queue = Queue("model_training_queue", limiter={"limit": 10, "period": 600}, concurrency=1)

# Initialize services
sentiment_model = SentimentModel()

# Create a single instance of MovieDatabase
movie_db = None

def get_movie_db():
    """Get or create MovieDatabase instance"""
    global movie_db
    if movie_db is None:
        movie_db = MovieDatabase()
    return movie_db

# Add user state tracking
USER_STATES = {}  # Stores the current state and context for each user
class UserState:
    INITIAL = "initial"
    SELECTING_MOVIE = "selecting_movie"
    RATING_MOVIE = "rating_movie"

# @app.post("/")
# def slack_challenge(request: FastAPIRequest, body: Dict[str, Any] = Body(...)):  # type: ignore
#     if "challenge" in body:
#         # Respond to the Slack challenge request
#         DBOS.logger.info("Received challenge")
#         return {"challenge": body["challenge"]}
#     # Dispatch other incoming requests to the Slack Bolt app
#     return slackapp.dispatch(to_bolt_request(request, request._body))

@app.post("/slack/events")
@DBOS.workflow()
async def handle_slack_event(request: FastAPIRequest, body: Dict[str, Any] = Body(...)):
    """Handle incoming Slack events"""
    if "challenge" in body:
        return {"challenge": body["challenge"]}

    event = body.get("event", {})
    if event.get("type") == "message":
        # Use event_id as idempotency key
        event_id = body.get("event_id", str(uuid.uuid4()))
        with SetWorkflowID(event_id):
            result = await process_slack_message(event)
            return {"ok": True, "result": result}
    return {"ok": True}

@DBOS.step()
async def process_slack_message(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process incoming Slack message"""
    user = event["user"]
    text = event["text"]
    channel = event["channel"]

    # Initialize user state if not exists
    if user not in USER_STATES:
        USER_STATES[user] = {"state": UserState.INITIAL, "context": {}}

    # Check if it's a greeting or process based on current state
    if is_greeting(text.lower()):
        USER_STATES[user] = {"state": UserState.SELECTING_MOVIE, "context": {}}
        await send_trending_movies(channel)
        return {"message": "Sent trending movies"}
    else:
        feedback_queue.enqueue(
            process_message,
            {"text": text, "user": user, "ts": event["ts"], "channel": channel}
        )
        return {"message": "Queued message processing"}

def is_greeting(text: str) -> bool:
    """Check if the message is a greeting"""
    greetings = ["oi", "olá", "hi", "hello", "hey", "start", "movies"]
    return any(greeting in text.lower() for greeting in greetings)

@DBOS.workflow()
async def send_trending_movies(channel: str):
    """Send trending movies to the channel"""
    trending_movies = await _get_and_format_trending_movies()
    await _send_slack_message(channel, trending_movies)

@DBOS.step()
async def _get_and_format_trending_movies() -> str:
    """Get and format trending movies message"""
    trending_movies = get_movie_db().get_trending_movies(limit=10)
    message = "Here are the top 10 trending movies:\n"
    for idx, movie in enumerate(trending_movies, 1):
        message += f"{idx} - {movie['title']}\n"
    return message

@DBOS.step()
async def _send_slack_message(channel: str, message: str):
    """Send message to Slack channel"""
    await slack_app.client.chat_postMessage(
        channel=channel,
        text=message
    )

@DBOS.workflow()
async def process_message(message_data: Dict[str, Any]):
    """Process user message based on current state"""
    text = message_data["text"]
    user = message_data["user"]
    channel = message_data["channel"]
    user_state = USER_STATES.get(user, {"state": UserState.INITIAL, "context": {}})

    if user_state["state"] == UserState.SELECTING_MOVIE:
        await _handle_movie_selection(text, user, channel, user_state)
    elif user_state["state"] == UserState.RATING_MOVIE:
        await _handle_movie_rating(text, user, channel, user_state)

@DBOS.step()
async def _handle_movie_selection(text: str, user: str, channel: str, user_state: Dict):
    """Handle movie selection state"""
    if text.isdigit() and 1 <= int(text) <= 10:
        movie_number = int(text)
        movie = get_movie_db().get_trending_movies(limit=10)[movie_number - 1]
        user_state["state"] = UserState.RATING_MOVIE
        user_state["context"]["current_movie"] = movie
        USER_STATES[user] = user_state
        await send_movie_details(channel, movie, user)
    else:
        await send_invalid_selection_message(channel)

@DBOS.step()
async def _handle_movie_rating(text: str, user: str, channel: str, user_state: Dict):
    """Handle movie rating state"""
    if text.isdigit() and 1 <= int(text) <= 5:
        rating = int(text)
        movie = user_state["context"].get("current_movie")
        if movie:
            sentiment = rating * 20
            store_user_reaction(user, str(rating), sentiment, movie["id"])

            user_state["state"] = UserState.SELECTING_MOVIE
            user_state["context"] = {}
            USER_STATES[user] = user_state

            await send_rating_confirmation(channel, rating, movie["title"])
    else:
        await send_invalid_rating_message(channel)

@DBOS.workflow()
async def send_movie_details(channel: str, movie: Dict, user: str):
    """Send movie details and prompt for rating"""
    await _check_and_send_movie_details(channel, movie, user)

@DBOS.step()
async def _check_and_send_movie_details(channel: str, movie: Dict, user: str):
    """Check if user watched movie and send appropriate message"""
    watched = get_movie_db().has_user_watched(movie['id'], user)

    message = f"*{movie['title']}*\n{movie['plot']}\n\n"
    if watched:
        message += "I've watched this movie! Please rate it (1-5 stars)!"
    else:
        message += "I haven't watched this one yet. Let's skip to another movie!"

    await slack_app.client.chat_postMessage(
        channel=channel,
        text=message
    )

@DBOS.workflow()
def store_user_reaction(user: str, text: str, sentiment: int, movie_id: str):
    """Store user reaction and potentially trigger recommendations"""
    _store_reaction_and_check(user, text, sentiment, movie_id)

@DBOS.step()
def _store_reaction_and_check(user: str, text: str, sentiment: int, movie_id: str):
    """Store reaction and check if recommendations should be triggered"""
    # Store the reaction
    get_movie_db().store_reaction(user, text, sentiment, movie_id)

    # Check if user has rated 5 movies
    if get_movie_db().get_user_ratings_count(user) >= 5:
        # Queue recommendation generation with idempotency
        movie_recommendation_queue.enqueue(
            send_recommendations,
            {"user": user},
            idempotency_key=f"recommendations_{user}_{movie_id}"
        )

@DBOS.workflow()
async def send_recommendations(data: Dict[str, str]):
    """Send movie recommendations to user"""
    await _generate_and_send_recommendations(data["user"])

@DBOS.step()
async def _generate_and_send_recommendations(user: str):
    """Generate and send personalized recommendations"""
    recommendations = get_movie_db().get_recommendations(user)
    message = "Based on your ratings, you might like:\n"
    for idx, movie in enumerate(recommendations, 1):
        message += f"{idx} - {movie['title']}\n"

    await slack_app.client.chat_postMessage(
        channel=user,
        text=message
    )

#Spot instance for training, keep offset from older reactions
@DBOS.scheduled("*/50 * * * *")  # Runs every 50 minutes
@DBOS.workflow()
def schedule_model_training(scheduled_time: datetime, actual_time: datetime):
    """Schedule periodic model training"""
    model_training_queue.enqueue(train_model)

@DBOS.workflow()
def train_model():
    """Train the sentiment model"""
    _perform_model_training()

@DBOS.step()
def _perform_model_training():
    """Perform the actual model training"""
    reactions = get_movie_db().get_all_reactions()
    sentiment_model.train(reactions)
    sentiment_model.push_to_hub()

@DBOS.workflow()
async def send_invalid_selection_message(channel: str):
    """Send invalid selection message"""
    await _send_invalid_selection(channel)

@DBOS.step()
async def _send_invalid_selection(channel: str):
    """Send message for invalid movie selection"""
    await slack_app.client.chat_postMessage(
        channel=channel,
        text="Please select a valid movie number between 1 and 10."
    )

@DBOS.workflow()
async def send_invalid_rating_message(channel: str):
    """Send invalid rating message"""
    await _send_invalid_rating(channel)

@DBOS.step()
async def _send_invalid_rating(channel: str):
    """Send message for invalid rating"""
    await slack_app.client.chat_postMessage(
        channel=channel,
        text="Please provide a rating between 1 and 5 stars."
    )

@DBOS.workflow()
async def send_rating_confirmation(channel: str, rating: int, movie_title: str):
    """Send rating confirmation and show trending movies"""
    await _send_confirmation_and_trending(channel, rating, movie_title)

@DBOS.step()
async def _send_confirmation_and_trending(channel: str, rating: int, movie_title: str):
    """Send confirmation message and trigger trending movies display"""
    await slack_app.client.chat_postMessage(
        channel=channel,
        text=f"Thanks for rating '{movie_title}' with {rating} stars! Here are the trending movies again:"
    )
    await send_trending_movies(channel)

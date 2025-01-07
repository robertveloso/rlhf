from slack_bolt import App, BoltRequest
from slack_bolt.adapter.starlette.handler import to_bolt_request

from fastapi import Body, FastAPI
from fastapi import Request as FastAPIRequest

from typing import Dict, Any, List
from dbos import DBOS, Queue, SetWorkflowID
import os
import datetime

from imdbbot.schemas.movies import MovieDatabase
from imdbbot.app.sentiment import SentimentModel

app = FastAPI()

slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    logger=DBOS.logger
)

feedback_queue = Queue("movie_feedback_queue", limiter={"limit": 300, "period": 60}, concurrency=1)
movie_recommendation_queue = Queue("movie_recommendation_queue", limiter={"limit": 300, "period": 60}, concurrency=1)
model_training_queue = Queue("model_training_queue", limiter={"limit": 10, "period": 600}, concurrency=1)
event_processing_queue = Queue("slack_event_queue", limiter={"limit": 300, "period": 60}, concurrency=1)

sentiment_model = SentimentModel()
movie_db = None

USER_STATES = {}
class UserState:
    INITIAL = "initial"
    SELECTING_MOVIE = "selecting_movie"
    RATING_MOVIE = "rating_movie"

@app.post("/slack/events")
def slack_events(request: FastAPIRequest, body: Dict[str, Any] = Body(...)):
    if "challenge" in body:
        DBOS.logger.info("Received challenge")
        return {"challenge": body["challenge"]}
    return slack_app.dispatch(to_bolt_request(request, request._body))

# @slack_app.message()
# def handle_message(request: BoltRequest) -> None:
#     DBOS.logger.info(f"Received message: {request.body}")
#     event_id = request.body["event_id"]

#     try:
#         with SetWorkflowID(event_id):
#             handle = event_processing_queue.enqueue(
#                 message_workflow,
#                 request.body["event"]
#             )
#             handle.get_result()
#     except Exception as e:
#         DBOS.logger.error(f"Error processing message: {e}")
@slack_app.message()
def handle_message(request: BoltRequest) -> None:
    DBOS.logger.info(f"Received message: {request.body}")
    event_id = request.body["event_id"]

    try:
        with SetWorkflowID(event_id):
            message_workflow(request.body["event"])
            return {"ok": True}
    except Exception as e:
        DBOS.logger.error(f"Error processing message: {e}")
        return {"ok": False}

@DBOS.workflow()
def message_workflow(message: Dict[str, Any]) -> None:
    DBOS.logger.info("Initializing movie database...")
    initialize_movie_db()
    DBOS.logger.info("Processing Slack message...")
    process_slack_message(message)

@DBOS.transaction()
def initialize_movie_db():
    """Get or create MovieDatabase instance"""
    MovieDatabase.initialize_database(DBOS.sql_session, "robertveloso/movie-sentiment")


@DBOS.step()
def process_slack_message(event: Dict[str, Any]):
    """Process incoming Slack message"""
    user = event["user"]
    text = event["text"]
    channel = event["channel"]

    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return

    if user not in USER_STATES:
        USER_STATES[user] = {"state": UserState.INITIAL, "context": {}}

    if is_greeting(text.lower()):
        USER_STATES[user] = {"state": UserState.SELECTING_MOVIE, "context": {}}
        send_trending_movies(channel)
    else:
        feedback_queue.enqueue(
            process_message,
            {"text": text, "user": user, "ts": event["ts"], "channel": channel}
        )

def is_greeting(text: str) -> bool:
    """Check if the message is a greeting"""
    greetings = ["oi", "olá", "hi", "hello", "hey", "start", "movies"]
    return any(greeting in text.lower() for greeting in greetings)

@DBOS.workflow()
async def send_trending_movies(channel: str):
    """Send trending movies to the channel"""
    trending_movies = await _get_and_format_trending_movies()
    _send_slack_message(channel, trending_movies)

@DBOS.step()
async def _get_and_format_trending_movies() -> str:
    """Get and format trending movies message"""
    trending_movies = await _get_trending_movies()

    if not trending_movies:
        trending_movies = await _get_initial_recommendations()

    message = "Here are the top 10 movies:\n"
    for idx, movie in enumerate(trending_movies, 1):
        message += f"{idx} - {movie['title']}\n"
    return message

@DBOS.transaction()
async def _get_trending_movies(limit: int = 10) -> List[Dict]:
    """Database transaction to get trending movies"""
    return MovieDatabase.get_trending_movies(DBOS.sql_session, limit=limit)

@DBOS.transaction()
async def _get_initial_recommendations(limit: int = 10) -> List[Dict]:
    """Database transaction to get initial recommendations"""
    return MovieDatabase.get_initial_recommendations(DBOS.sql_session, limit=limit)

@DBOS.step()
def _send_slack_message(channel: str, message: str):
    """Send message to Slack channel"""
    slack_app.client.chat_postMessage(
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

@DBOS.transaction()
async def _handle_movie_selection(text: str, user: str, channel: str, user_state: Dict):
    """Handle movie selection state"""
    if text.isdigit() and 1 <= int(text) <= 10:
        movie_number = int(text)
        movie = MovieDatabase.get_trending_movies(DBOS.sql_session, limit=10)[movie_number - 1]
        user_state["state"] = UserState.RATING_MOVIE
        user_state["context"]["current_movie"] = movie
        USER_STATES[user] = user_state
        await send_movie_details(channel, movie, user)
    else:
        await send_invalid_selection_message(channel)

@DBOS.transaction()
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

@DBOS.transaction()
async def _check_and_send_movie_details(channel: str, movie: Dict, user: str):
    """Check if user watched movie and send appropriate message"""
    watched = MovieDatabase.has_user_watched(DBOS.sql_session, movie['id'], user)

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

@DBOS.transaction()
def _store_reaction_and_check(user: str, text: str, sentiment: int, movie_id: str):
    """Store reaction and check if recommendations should be triggered"""
    MovieDatabase.store_reaction(DBOS.sql_session, user, text, sentiment, movie_id)

    if MovieDatabase.get_user_ratings_count(DBOS.sql_session, user) >= 5:
        movie_recommendation_queue.enqueue(
            send_recommendations,
            {"user": user},
            idempotency_key=f"recommendations_{user}_{movie_id}"
        )

@DBOS.workflow()
async def send_recommendations(data: Dict[str, str]):
    """Send movie recommendations to user"""
    await _generate_and_send_recommendations(data["user"])

@DBOS.transaction()
async def _generate_and_send_recommendations(user: str):
    """Generate and send personalized recommendations"""
    recommendations = MovieDatabase.get_recommendations(DBOS.sql_session, user)
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

@DBOS.transaction()
def _perform_model_training():
    """Perform the actual model training"""
    reactions = MovieDatabase.get_all_reactions(DBOS.sql_session)
    sentiment_model.train(reactions)
    sentiment_model.push_to_hub()

@DBOS.workflow()
async def send_invalid_selection_message(channel: str):
    """Send invalid selection message"""
    await _send_invalid_selection(channel)

@DBOS.transaction()
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
    slack_app.client.chat_postMessage(
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
    trending_movies = await _get_and_format_trending_movies()
    confirmation_message = f"Thanks for rating '{movie_title}' with {rating} stars!\n\n{trending_movies}"

    slack_app.client.chat_postMessage(
        channel=channel,
        text=confirmation_message
    )

from typing import Dict, List, Any
from datasets import load_dataset
from dbos import DBOS
from sqlalchemy import (
    Table, Column, String, Integer, Float, BigInteger, ForeignKey,
    select, func, text, MetaData
)
from sqlalchemy.sql import and_, not_
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re

metadata = MetaData()

movies = Table(
    "movies",
    metadata,
    Column("id", String, primary_key=True),
    Column("title", String, nullable=False),
    Column("plot", String),
    Column("rating", Float),
    Column(
        "created_at",
        BigInteger,
        nullable=False,
        server_default=text("(EXTRACT(epoch FROM now()) * 1000::numeric)::bigint")
    ),
)

user_reactions = Table(
    "user_reactions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, nullable=False),
    Column("movie_id", String, ForeignKey("movies.id"), nullable=False),
    Column("rating", Integer),
    Column("sentiment", Integer),
    Column(
        "created_at",
        BigInteger,
        nullable=False,
        server_default=text("(EXTRACT(epoch FROM now()) * 1000::numeric)::bigint")
    ),
)

class MovieDatabase:
    def initialize_database(session, model_id):
        """Initialize the database with IMDB dataset and pre-trained model recommendations"""
        count = session.scalar(select(func.count()).select_from(movies))
        if count == 0:
            dataset = load_dataset("imdb", split="train")

            # Load pre-trained model
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            # Process movies in batches
            batch_size = 32
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]

                titles = []
                for text in batch['text']:
                    first_line = text.split('\n')[0]

                    title = "Unknown Movie"
                    # Try to find quoted titles first
                    title_match = re.search(r'"([^"]+)"|\'([^\']+)\'', first_line)
                    if title_match:
                        title = title_match.group(1) or title_match.group(2)
                    else:
                        # Try to find capitalized words as fallback
                        caps_match = re.search(r'([A-Z][A-Za-z0-9\s&:-]+(?=\s|$))', first_line)
                        if caps_match:
                            title = caps_match.group(1).strip()

                    titles.append(title)

                # Get sentiment scores for each movie
                with torch.no_grad():
                    inputs = tokenizer(
                        batch['text'],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512
                    ).to(device)
                    outputs = model(**inputs)
                    scores = torch.softmax(outputs.logits, dim=1)

                for idx, (text, score, title) in enumerate(zip(batch['text'], scores, titles)):
                    positive_score = score[1].item()
                    if positive_score >= 0.8:
                        stmt = movies.insert().values(
                            id=f"imdb_{i+idx}",
                            title=title,
                            plot=text[:500],
                            rating=positive_score
                        )
                        session.execute(stmt)

    def get_trending_movies(session, limit: int = 10) -> List[Dict]:
        """Get trending movies based on recent user reactions"""
        stmt = (
            select(
                movies.c.id,
                movies.c.title,
                movies.c.plot,
                func.avg(user_reactions.c.rating).label('avg_rating')
            )
            .select_from(movies.outerjoin(user_reactions))
            .group_by(movies.c.id, movies.c.title, movies.c.plot)
            .order_by(func.count(user_reactions.c.id).desc(), text('avg_rating DESC'))
            .limit(limit)
        )
        result = session.execute(stmt)

        return [
            {
                'id': row.id,
                'title': row.title,
                'plot': row.plot,
                'rating': float(row.avg_rating) if row.avg_rating else 0
            }
            for row in result
        ]

    def has_user_watched(session, movie_id: str, user_id: str) -> bool:
        """Check if user has watched a movie"""
        stmt = select(func.count()).select_from(user_reactions).where(
            and_(
                user_reactions.c.movie_id == movie_id,
                user_reactions.c.user_id == user_id
            )
        )
        return session.scalar(stmt) > 0

    def store_reaction(session, user_id: str, text: str, sentiment: int, movie_id: str):
        """Store user reaction to a movie"""
        stmt = user_reactions.insert().values(
            user_id=user_id,
            movie_id=movie_id,
            rating=int(text) if text.isdigit() else None,
            sentiment=sentiment
        )
        session.execute(stmt)

    def get_user_ratings_count(session, user_id: str) -> int:
        """Get number of movies rated by user"""
        stmt = select(func.count()).select_from(user_reactions).where(
            and_(
                user_reactions.c.user_id == user_id,
                user_reactions.c.rating.isnot(None)
            )
        )
        return session.scalar(stmt)

    def get_recommendations(session, user_id: str, limit: int = 5) -> List[Dict]:
        """Get movie recommendations based on user's ratings"""
        user_avg = (
            select(func.avg(user_reactions.c.rating).label('avg_rating'))
            .where(
                and_(
                    user_reactions.c.user_id == user_id,
                    user_reactions.c.rating.isnot(None)
                )
            )
            .cte('user_avg')
        )

        stmt = (
            select(movies.c.id, movies.c.title, movies.c.plot)
            .distinct()
            .select_from(
                movies
                .join(user_reactions)
                .join(user_avg)
            )
            .where(
                and_(
                    user_reactions.c.rating > user_avg.c.avg_rating,
                    not_(
                        movies.c.id.in_(
                            select(user_reactions.c.movie_id)
                            .where(user_reactions.c.user_id == user_id)
                        )
                    )
                )
            )
            .order_by(movies.c.rating.desc())
            .limit(limit)
        )

        result = session.execute(stmt)
        return [
            {'id': row.id, 'title': row.title, 'plot': row.plot}
            for row in result
        ]

    def get_all_reactions(session) -> List[Dict]:
        """Get all user reactions for model training"""
        stmt = (
            select(
                user_reactions.c.user_id,
                user_reactions.c.movie_id,
                movies.c.title,
                user_reactions.c.rating,
                user_reactions.c.sentiment
            )
            .select_from(
                user_reactions.join(movies)
            )
            .where(user_reactions.c.rating.isnot(None))
        )

        result = session.execute(stmt)
        return [
            {
                'user_id': row.user_id,
                'movie_id': row.movie_id,
                'title': row.title,
                'rating': row.rating,
                'sentiment': row.sentiment
            }
            for row in result
        ]

    def get_initial_recommendations(session, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get initial movie recommendations using pre-trained model scores.

        Args:
            limit: Maximum number of movies to return

        Returns:
            List of movie dictionaries containing 'id', 'title', and 'plot'
        """
        stmt = (
            select(movies.c.id, movies.c.title, movies.c.plot)
            .select_from(movies)
            .order_by(movies.c.rating.desc())
            .limit(limit)
        )

        result = session.execute(stmt)
        return [
            {
                "id": row.id,
                "title": row.title,
                "plot": row.plot
            }
            for row in result
        ]

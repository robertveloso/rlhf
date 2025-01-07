from typing import Dict, List
from datasets import load_dataset
from dbos import DBOS
from sqlalchemy import (
    Table, Column, String, Integer, Float, BigInteger, ForeignKey,
    select, func, text, MetaData
)
from sqlalchemy.sql import and_, not_

# Define SQLAlchemy metadata
metadata = MetaData()

# Define SQLAlchemy tables
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
    def __init__(self):
        """Initialize MovieDatabase with DBOS session"""
        self.session = DBOS.sql_session

    @DBOS.workflow()
    def initialize_database(self):
        """Initialize the database with IMDB dataset"""
        self._load_imdb_dataset()

    @DBOS.step()
    def _load_imdb_dataset(self):
        """Load IMDB dataset if movies table is empty"""
        count = self.session.scalar(select(func.count()).select_from(movies))

        if count == 0:
            dataset = load_dataset("imdb", split="train")
            for idx, item in enumerate(dataset):
                stmt = movies.insert().values(
                    id=f"imdb_{idx}",
                    title=f"Movie {idx}",
                    plot=item['text'][:200],
                    rating=float(item['label'])
                )
                self.session.execute(stmt)

    @DBOS.workflow()
    def get_trending_movies(self, limit: int = 10) -> List[Dict]:
        """Get trending movies based on recent user reactions"""
        return self._fetch_trending_movies(limit)

    @DBOS.step()
    def _fetch_trending_movies(self, limit: int) -> List[Dict]:
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
        result = self.session.execute(stmt)

        return [
            {
                'id': row.id,
                'title': row.title,
                'plot': row.plot,
                'rating': float(row.avg_rating) if row.avg_rating else 0
            }
            for row in result
        ]

    @DBOS.workflow()
    def has_user_watched(self, movie_id: str, user_id: str) -> bool:
        """Check if user has watched a movie"""
        return self._check_user_watched(movie_id, user_id)

    @DBOS.step()
    def _check_user_watched(self, movie_id: str, user_id: str) -> bool:
        stmt = select(func.count()).select_from(user_reactions).where(
            and_(
                user_reactions.c.movie_id == movie_id,
                user_reactions.c.user_id == user_id
            )
        )
        return self.session.scalar(stmt) > 0

    @DBOS.workflow()
    def store_reaction(self, user_id: str, text: str, sentiment: int, movie_id: str):
        """Store user reaction to a movie"""
        self._save_reaction(user_id, text, sentiment, movie_id)

    @DBOS.step()
    def _save_reaction(self, user_id: str, text: str, sentiment: int, movie_id: str):
        stmt = user_reactions.insert().values(
            user_id=user_id,
            movie_id=movie_id,
            rating=int(text) if text.isdigit() else None,
            sentiment=sentiment
        )
        self.session.execute(stmt)

    @DBOS.workflow()
    def get_user_ratings_count(self, user_id: str) -> int:
        """Get number of movies rated by user"""
        return self._fetch_user_ratings_count(user_id)

    @DBOS.step()
    def _fetch_user_ratings_count(self, user_id: str) -> int:
        stmt = select(func.count()).select_from(user_reactions).where(
            and_(
                user_reactions.c.user_id == user_id,
                user_reactions.c.rating.isnot(None)
            )
        )
        return self.session.scalar(stmt)

    @DBOS.workflow()
    def get_recommendations(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get movie recommendations based on user's ratings"""
        return self._fetch_recommendations(user_id, limit)

    @DBOS.step()
    def _fetch_recommendations(self, user_id: str, limit: int) -> List[Dict]:
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

        result = self.session.execute(stmt)
        return [
            {'id': row.id, 'title': row.title, 'plot': row.plot}
            for row in result
        ]

    @DBOS.workflow()
    def get_all_reactions(self) -> List[Dict]:
        """Get all user reactions for model training"""
        return self._fetch_all_reactions()

    @DBOS.step()
    def _fetch_all_reactions(self) -> List[Dict]:
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

        result = self.session.execute(stmt)
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

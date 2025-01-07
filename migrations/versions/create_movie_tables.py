"""
Initialize movie database tables.

Revision ID: [your_revision_id]
Revises:
Create Date: [current_date_time]
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "[your_revision_id]"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create movies table
    op.create_table(
        "movies",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("plot", sa.String),
        sa.Column("rating", sa.Float),
        sa.Column(
            "created_at",
            sa.BigInteger,
            nullable=False,
            server_default=sa.text(
                "(EXTRACT(epoch FROM now()) * 1000::numeric)::bigint"
            ),
        ),
    )

    # Create user reactions table
    op.create_table(
        "user_reactions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String, nullable=False),
        sa.Column("movie_id", sa.String, nullable=False),
        sa.Column("rating", sa.Integer),
        sa.Column("sentiment", sa.Integer),
        sa.Column(
            "created_at",
            sa.BigInteger,
            nullable=False,
            server_default=sa.text(
                "(EXTRACT(epoch FROM now()) * 1000::numeric)::bigint"
            ),
        ),
        sa.ForeignKeyConstraint(["movie_id"], ["movies.id"]),
    )

def downgrade() -> None:
    op.drop_table("user_reactions")
    op.drop_table("movies")
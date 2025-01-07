import praw
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime


class RedditMovieCrawler:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.environ['REDDIT_CLIENT_ID'],
            client_secret=os.environ['REDDIT_CLIENT_SECRET'],
            user_agent="MovieSentimentBot/1.0"
        )

    def crawl_movie_comments(self, movies, subreddits=['filmeseseries', 'filmes', 'filmesbr'], limit=50):
        data = []

        for movie in movies:
            print(f"Searching comments for '{movie}' in subreddits: {subreddits}")
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)

                for submission in tqdm(subreddit.search(movie, limit=limit, sort='comments')):
                    submission.comments.replace_more(limit=0)

                    comments = [
                        {
                            'text': comment.body,
                            'score': comment.score,
                            'subreddit': subreddit_name,
                            'movie': movie,
                            'created_utc': comment.created_utc
                        }
                        for comment in submission.comments.list()
                        if len(comment.body) <= 255  # Ensure comment length is 255 or less
                    ]

                    # Get the top 10 highest-scoring comments
                    sorted_comments = sorted(comments, key=lambda x: x['score'], reverse=True)[:10]
                    data.extend(sorted_comments)

        # Save to CSV
        df = pd.DataFrame(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reddit_data_{timestamp}.csv"

        df.to_csv(filename, index=False)
        print(f"Saved {len(data)} comments to {filename}")

        return df


if __name__ == "__main__":
    movies = ['Auto da Compadecida', 'Auto da Compadecida 2', 'Minha mãe é uma peça']
    crawler = RedditMovieCrawler()
    crawler.crawl_movie_comments(movies)

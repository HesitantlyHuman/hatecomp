from tqdm import tqdm
import logging
import aiometer  # TODO dont import this if not necessary
import asyncio
import httpx
import tweepy
import json
import functools
import random

TWITTER_DOMAIN = "api.twitter.com"


class TweetDownloader:
    def __init__(self):
        self.revision = 0
        self.KEYS = [
            ("IQKbtAYlXLripLGPWd0HUA", "GgDYlkSvaPxGxC4X8liwpUoqKwwr3lCADbz8A7ADU"),
            ("CjulERsDeqhhjSme66ECg", "IQWdVyqFxghAtURHGeGiWAsmCAGmdW3WmbEx6Hck"),
            ("3rJOl1ODzm9yZy63FACdg", "5jPoQ5kQvMJFDYRNE8bQ4rHuds4xJqhvgNJM4awaE8"),
            ("3nVuSoBZnx6U4vzUxf5w", "Bcs59EFbbsdF6Sl9Ng71smgStWEGwXXKSjYvPVt7qys"),
        ]
        random.shuffle(self.KEYS)
        self.keys = self.KEYS
        self.get_authorization()
        self.endpoint = f"https://{TWITTER_DOMAIN}/1.1/statuses/lookup.json"

    def download(self, tweet_ids):
        return asyncio.run(self.download_async(tweet_ids))

    async def download_async(self, tweet_ids):
        print("Downloading tweets...")
        tweets = {}
        progress_bar = tqdm(total=len(tweet_ids))
        async with httpx.AsyncClient(timeout=30) as session:
            chunks = TweetDownloader.chunk_list(tweet_ids, 100)
            async with aiometer.amap(
                functools.partial(self._download_helper, session, progress_bar),
                chunks,
                max_at_once=5,
                max_per_second=2,
            ) as chunk_results:
                async for tweet_results in chunk_results:
                    for tweet_id, tweet_content in tweet_results:
                        tweets[tweet_id] = tweet_content
        return tweets

    async def _download_helper(self, session, progress_bar, chunk):
        revision = self.revision
        try:
            tweets = await self.download_batch(session, chunk)
        except TooFast:
            logging.info("TweetDownloader exceeded rate limits")
            if revision == self.revision:
                logging.info("Switching authorization")
                self.get_authorization()
        tweets = await self.download_batch(session, chunk)
        progress_bar.update(len(chunk))
        return tweets

    async def download_batch(self, session, tweet_ids):
        input_type = type(tweet_ids[0])
        query = {"id": ",".join([str(tweet_id) for tweet_id in tweet_ids])}
        response = await session.get(
            url=self.endpoint,
            headers=self.headers,
            params=query,
        )
        api_data = json.loads(response.text)
        try:
            return [
                (input_type(tweet_data["id"]), tweet_data["text"])
                for tweet_data in api_data
            ]
        except TypeError as e:
            try:
                if (
                    "rate limit" in api_data["errors"][0]["message"].lower()
                    or "88" in api_data["errors"][0]["code"]
                ):
                    raise TooFast
                else:
                    raise e
            except Exception as e:
                raise e

    def chunk_list(list_in, chunk_size):
        chunks = []
        num_chunks = len(list_in) // chunk_size
        for i in range(num_chunks):
            chunks.append(list_in[i * chunk_size : i * chunk_size + chunk_size])
        chunks.append(list_in[(num_chunks) * chunk_size :])
        return chunks

    def get_authorization(self):
        if len(self.keys) == 0:
            self.keys = self.KEYS
        key, secret = self.keys.pop()
        self.auth = tweepy.AppAuthHandler(key, secret)
        logging.info(f"TweetDownloader using Twitter API Key: {key}")
        self.headers = {
            "Host": TWITTER_DOMAIN,
            "User-Agent": "python-requests/2.26.0",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "*/*",
            "Connection": "keep-alive",
            "Authorization": f"Bearer {self.auth._bearer_token}",
        }
        self.revision += 1

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass


class TooFast(Exception):
    pass


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    downloader = TweetDownloader()
    print(
        downloader.download(
            [
                "572348198062170112",
            ]
        )
    )

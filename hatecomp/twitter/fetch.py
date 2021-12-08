import os
import json
import aiohttp
import asyncio

try:
    TWITTER_KEY_PATH = os.environ['TWITTER_KEYS']
except KeyError:
    TWITTER_KEY_PATH = 'hatecomp/twitter/keys.json'
    print(f"TWITTER_KEYS environment variable not set, using default location of {TWITTER_KEY_PATH}")

class TweetDownloader():
    def __init__(self, twitter_keys_path = TWITTER_KEY_PATH):
        with open(twitter_keys_path) as twitter_keys:
            self.keys = json.loads(twitter_keys.read())
        self.endpoint = "https://api.twitter.com/2/tweets/{}"
        self.headers = {
            "Authorization" : "Bearer " + self.keys['Bearer Token']
        }

    def download(self, tweet_ids):
        return asyncio.run(self.download_async(tweet_ids))

    async def download_async(self, tweet_ids):
        self.session = aiohttp.ClientSession()
        coroutines = [self.get_tweet(tweet_id) for tweet_id in tweet_ids]
        content = await asyncio.gather(*coroutines)
        await self.session.close()
        return content

    async def get_tweet(self, tweet_id):
        result =  await self.session.get(
            url = self.endpoint.format(tweet_id),
            headers = self.headers
        )
        tweet = await result.text()
        try:
            return json.loads(tweet)['data']['text']
        except KeyError:
            return None

    async def _get_session(self, keys) -> aiohttp.ClientSession:
        return aiohttp.ClientSession()
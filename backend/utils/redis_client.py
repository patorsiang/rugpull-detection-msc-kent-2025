import os
import redis
from dotenv import load_dotenv

load_dotenv()

class RedisNamespace:
    def __init__(self, host=None, port=None, db=0, namespace="default"):
        self.redis = redis.Redis(
            host=host or os.getenv("REDIS_HOST", "localhost"),
            port=port or int(os.getenv("REDIS_PORT", 6379)),
            db=db,
            decode_responses=True
        )
        self.ns = namespace

    def _key(self, key):
        return f"{self.ns}:{key}"

    def set(self, key, value, ex=None):
        return self.redis.set(self._key(key), value, ex=ex)

    def get(self, key):
        return self.redis.get(self._key(key))

    def delete(self, key):
        return self.redis.delete(self._key(key))

    def lpush(self, key, value):
        return self.redis.lpush(self._key(key), value)

    def lrange(self, key, start, end):
        return self.redis.lrange(self._key(key), start, end)

    def hset(self, key, mapping):
        return self.redis.hset(self._key(key), mapping=mapping)

    def hgetall(self, key):
        return self.redis.hgetall(self._key(key))


# Instances
redis_cache = RedisNamespace(db=0, namespace="topic")             # topic cache
redis_log   = RedisNamespace(db=1, namespace="log:predict")       # prediction cache
redis_flags = RedisNamespace(db=2, namespace="lock")              # trigger flags / locks
redis_meta  = RedisNamespace(db=3, namespace="meta")              # metadata

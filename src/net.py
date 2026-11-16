"""Kibar HTTP istemcisi: rastgele bekleme, disk onbellegi, tekrar deneme.

Ayni URL bir kez indirilir ve data/cache altinda saklanir; bitmis maclar
icin siteye ikinci kez gidilmez.
"""
import hashlib
import json
import random
import time

import requests

from config import CACHE, MAX_DELAY, MIN_DELAY, USER_AGENT

_last_request = 0.0


def _wait():
    global _last_request
    gap = random.uniform(MIN_DELAY, MAX_DELAY)
    elapsed = time.time() - _last_request
    if elapsed < gap:
        time.sleep(gap - elapsed)
    _last_request = time.time()


def _cache_path(source, url):
    key = hashlib.sha1(url.encode()).hexdigest()
    return CACHE / source / f"{key}.json"


def get_json(url, source, headers=None, use_cache=True, store=True, impersonate=False, retries=4):
    """URL'den JSON getirir. use_cache=False ise taze ceker; store=False ise diske yazmaz."""
    path = _cache_path(source, url)
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        h.update(headers)

    for attempt in range(retries):
        _wait()
        try:
            if impersonate:
                from curl_cffi import requests as cr
                r = cr.get(url, headers=h, impersonate="chrome", timeout=30)
            else:
                r = requests.get(url, headers=h, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429, 503):
                # Engel ya da hiz siniri: uzun bekle
                time.sleep(60 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            if not store:
                return data
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except Exception as e:  # requests, curl_cffi ya da bozuk JSON
            print(f"  ! {source} hata ({attempt + 1}/{retries}): {e}")
            time.sleep(2 ** (attempt + 1))
    return None

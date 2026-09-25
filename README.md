 # RecZilla

 ## Quick Start Guide
    - Requires Python 3.10+
    - Clone the repo: https://github.com/MajorTomLanded/rec-zilla.git
    - (recommended) create and activate a virtualenv: `python3 -m venv .venv && source .venv/bin/activate`
    - install dependencies: `pip install -r requirements.txt`

    ### Components 

    - `model/data_client.py`: downloads, merges, and cleans data.
    - `model/recommender.py`: builds item to item similarity scores, core recommend functionality
    - `model/cold_start.py`:  build active user embeddings, seeds new user ratings data
    - `main.py`: CLI interface for recommender and cold start

    ### Get a recommendation for an existing user

    No API key needed for this; the course + MovieLens data downloads automatically on first run.

    `python main.py recommend course_1 -n 10`
    - replace "course_1" with any user ID you want to try from the dataset.

    ### Cold start (new user, free-text description) - requires a Gemini API key

    - Get a free Gemini API key: https://ai.google.dev/gemini-api/docs/api-key
    - Copy `.env.example` to `.env` and set `GOOGLE_API_KEY="<your-key-here>"`
    - First-time warm-up embeds every active user's self-description via the Gemini API (currently ~750 users) and caches the result to disk in `data/warm_embeddings.csv`, so it only needs to happen once. Expect it to take a while and possibly hit free-tier rate limits (the code retries automatically, up to 5 times, with a 60s backoff each time). The results are saved in batches and you can retry to pickup where you left off.

    `python main.py warm-up`

    - Once warmed up, get cold-start recommendations:

    `python main.py cold-start "I like long walks on the beach and sitting by the fireplace. I don't like conflict."`

    - `--cached-only` skips fetching embeddings for any users not already cached, useful if you're rate limited and want to proceed with a smaller (but real) neighbor pool. Only use this after `warm-up` generated some active user embeddings.

    `python main.py cold-start "I like long walks on the beach and sitting by the fireplace. I don't like conflict." --cached-only`
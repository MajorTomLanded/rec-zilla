 # RecZilla

 ## Quick Start Guide
    - Clone the repo: https://github.com/MajorTomLanded/rec-zilla.git

    - install dependencies: `pip install -r requirements.txt`
       
    ### Components 

    - `model/data_client.py`: downloads, merges, and cleans data.
    - `model/recommender.py`: builds item to item similarity scores, core recommend functionality
    - `model/cold_start.py`:  build active user embeddings, seeds new user ratings data
    - `main.py`: CLI interface for recommender and cold start

    ### How to use

    - after installing dependencies, add your Gemini API key to .env:
    `GOOGLE_API_KEY="<your-key-here>"`
    - .env.example has the template for the config
    - To warm-up the cache, run this in the terminal:
    `python main.py warm-up`
    - That is getting the embeddings for the active users in batches, it can take several minutes. The embeddings are saved on disk so you only need to do this once.
    - This command gets a recommendation for an existing user:
    `python main.py recommend course_1 -n 10`
    - replace "course_1" with any user ID you want to try from the dataset. 
    - To get cold-start recommendations, try this:
    `python main.py cold-start "I like long walks on the beach and sitting by the fireplace. I don't like conflict."`
    - Use --cached-only if you don't want to get net-new embeddings for users
    `python main.py cold-start "I like long walks on the beach and sitting by the fireplace. I don't like conflict." --cached-only`
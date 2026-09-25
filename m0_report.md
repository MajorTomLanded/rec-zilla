# M0 Report

## Learning 

    ### Data used

        I selected to use the data provided by the course, and augmented it with an additional interaction dataset from MovieLens (1). The MovieLens dataset used is a curated, high-quality dataset of real user movie ratings. I merged the course-provided ratings data with the MovieLens rating data, de-duplicating the movies using the TMDB id. The benefits of using these data sources are that the curated data will be cleaner and easier to work with, and these sources are free. The merged data will give me more interaction per movie with which to find item-to-item similarities. The trade off is that curated datasets are less true to life, and overfitting to them can create poor results against live data. I chose not to use IMDB non-commerical dataset and TMDB API because these sources don't provide interaction data. Richer movie data would be more useful if I was using an attribute-based solution. 

    ### Machine Learning technique

        I chose to use an item-to-item collaborative filtering technique (2) because it captures real life behaviors and more subtle similarities in a way using attributes doesn't. I chose movie-to-movie and not user-to-user because the movies across datasets can be deduplicated, increasing overlapping interactions. Users can not be deduplicated in order to create a higher concentration of reviews per user.

        One of the limitations of collaborative filtering is that sparse data makes it difficult to find enough neighbors. This dataset has 1.68% density - for rarer movies, there will be too few overlaping interactions. One way to overcome this limitation would be to use a hybrid approach (3). For movies with less density, we could fall back on content-based similarity scores. With more time, this is what I would build next.
    
    ### The cold start problem

        To solve the cold start problem, I want to give the new user a "persona" that seeds the interaction data with synthetic data based on the user input. This solution uses the same item to item system I've already created by passing in seeded ratings to the interaction matrix.
    
        To get synthetic "persona" data, I gathered the active users' self-descriptions and using Gemini's embedding model, I built vectors that I can use to find neighbors (4, 5) for the new user (based on the new user's self description). I used cosine similarity on this embeddings data - same approach that I use to find item to item similarity based on user ratings. Then I seed the user's interaction data with their closest neighbors' interaction data.

        Ideally this persona system would take feedback and improve over time, learning from cold-start users over time. This was not implemented with this deliverable, but I think it would be an important part of using a system like this in production. There were many important features considered but not yet implemented: provenance for the seeded data so it can be retired after the user generated enough interactions; An LLM-based genre and movie title extractor from the free text - this could have been a valuable input to a hybrid collaborative filtering + content-based recommender.

    ## Implementation Overview
    
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

CITATIONS:

1. Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19. https://doi.org/10.1145/2827872

2. Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th International Conference on World Wide Web (WWW '01), 285–295. https://doi.org/10.1145/371920.372071

3. Burke, R. (2002). Hybrid recommender systems: Survey and experiments. User Modeling and User-Adapted Interaction, 12(4), 331–370. https://doi.org/10.1023/A:1021240730564

4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of EMNLP-IJCNLP 2019, 3982–3992. https://arxiv.org/abs/1908.10084

5. Lee, J., et al. (2025). Gemini Embedding: Generalizable embeddings from Gemini. arXiv:2503.07891. https://arxiv.org/abs/2503.07891

AI usage disclosure: I used Claude (Anthropic) as a research and planning aide throughout this project. This included: explaining domain concepts, reviewing/debugging code I wrote, and scaffolding code from decisions I made myself. All modeling decisions, the report's analysis and conclusions, and the final code are my own; no report text was generated by AI and copied in verbatim.
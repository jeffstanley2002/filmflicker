# FilmFlicker Model Training and Evaluation Report

## Document purpose

This document is the permanent technical record of the FilmFlicker recommendation models, last rebuilt and evaluated on July 24, 2026. It explains the process from raw data preparation through production artifact validation.

It is intended to answer all of the following without requiring the reader to reverse-engineer the code:

- Which data was used?
- How was the data represented?
- What does each recommender actually do?
- Which parts are trained and which parts are rule-based?
- How are new FilmFlicker users handled when they do not exist in MovieLens?
- How were data leakage and test-set tuning prevented?
- Which parameters were tested?
- Why was the final configuration selected?
- What are the final accuracy, ranking, diversity, novelty, and coverage results?
- What does each metric mean in practical terms?
- Which models should be used for which product roles?
- What remains unproven or limited?
- How can the complete process be reproduced?

The source-of-truth machine-readable outputs are:

- `models/tuning_results.json`: validation search protocol, candidates, and winner
- `models/metrics.json`: committed chronological evaluation
- `data/processed/catalog_manifest.json`: processed data provenance

## Executive summary

FilmFlicker contains five recommendation strategies:

1. Bayesian popularity
2. TF-IDF content matching
3. Regularized collaborative matrix factorization
4. KMeans taste neighborhoods
5. Two-tower neural taste embeddings

These candidate generators share one production ensemble and reranker. The full-32M evaluation now shows content-based matching as the strongest top-10 ranker, while the collaborative hybrid remains the strongest star-rating predictor and most diverse personalized model. Each For You tab is served by its selected primary generator when possible. Popularity is the explicit cold-start lane, content matching also powers `more like this`, clustering powers interpretable discovery neighborhoods, and taste embeddings remain a comparison model.

The models were trained on MovieLens 32M:

| Data property | Value |
| --- | ---: |
| Movies | 87,585 |
| Ratings | 32,000,204 |
| Tags | 2,000,072 |
| Earliest parsed movie year | 1874 |
| Latest catalog year | 2023 |
| TMDB enrichment during this build | Disabled |

The final collaborative configuration was selected on a chronological validation set and evaluated on a separate chronological test set. Production collaborative, popularity, content, clustering, and taste-embedding artifacts were then trained from the full catalog inputs. The taste-embedding artifact is independently trained by a two-tower neural model on all 32,000,204 ratings; validation rejects the old SVD-derived fallback.

Current committed primary-model results use the full MovieLens 32M ratings set with 1,000 untouched user histories sampled from the chronological holdout.

| Metric | Best model | Value |
| --- | --- | ---: |
| RMSE | Collaborative hybrid | 0.8993 |
| MAE | Collaborative hybrid | 0.6716 |
| Precision@10 | Content-based | 1.50% |
| Recall@10 | Content-based | 4.30% |
| Hit Rate@10 | Content-based | 12.50% |
| NDCG@10 | Content-based | 3.28% |
| MRR@10 | Content-based | 5.19% |
| Intra-list diversity | Collaborative hybrid | 86.56% |
| Mean novelty | Taste neighborhoods | 21.83 bits |
| Catalog coverage | Content-based | 2.30% |

The deployment conclusion is:

- Ready for a resume/portfolio deployment with reproducible offline evidence.
- Ready to use content-based matching as the strongest top-10 recommendation tab.
- Ready to use collaborative as the best rating predictor and a strong diversity-focused personalized tab.
- Ready to use popularity as the cold-start/fallback model.
- Ready to use clustering for insights, not as the primary ranker.
- Not commercially proven because there is no real-user A/B testing, engagement data, or current post-2023 catalog enrichment.

## 1. Data provenance and preparation

### 1.1 Source dataset

The processed catalog was built from `data/ml-32m`, the MovieLens 32M dataset. The catalog manifest was generated on July 13, 2026.

The production training pipeline reads:

- `data/processed/movies.csv`
- `data/processed/ratings.csv`
- `data/processed/tags.csv`
- `data/processed/links.csv`

The environment variable `FILMFLICKER_DATA_DIR` can override this location. If no override is provided, the code prefers `data/processed` when a processed movie catalog exists and otherwise falls back to `data/ml-latest-small`.

### 1.2 Rating data types

Ratings are loaded with explicit compact types:

- `userId`: signed 32-bit integer
- `movieId`: signed 32-bit integer
- `rating`: 32-bit floating point
- `timestamp`: signed 64-bit integer

This materially reduces memory use when loading 32 million rows and keeps full-data retraining practical on a development machine.

### 1.3 Movie normalization

The movie loader:

- Parses the release year from a trailing `(YYYY)` when necessary.
- Converts the pipe-separated genre field into `genre_list`.
- Represents `(no genres listed)` as an empty list.
- Adds nullable poster and metadata columns when they are absent.
- Preserves MovieLens IDs as the canonical model/catalog join key.

### 1.4 Tags and text corpus

For content training, one text document is created per movie:

- Every genre token is repeated three times to provide a stable genre signal.
- User tags are lowercased.
- Spaces within tags are converted to underscores.
- All tags for the same movie are concatenated.
- The genre text and tag text are combined into one document.

The result is indexed by `movieId`, preserving a direct relationship between each sparse TF-IDF row and the catalog movie.

### 1.5 Catalog limitations

This artifact build did not use TMDB:

- `tmdb_enabled`: false
- `tmdb_enriched_movies`: 0
- `tmdb_current_movies_added`: 0

Consequences:

- The training catalog ends in 2023.
- Very recent releases are absent.
- Content features depend primarily on MovieLens genres and community tags rather than rich plot, cast, director, and keyword metadata.
- This is acceptable for the portfolio demonstration but should be disclosed.

## 2. Shared product and serving behavior

The app does not simply return the raw top ten from a single model. Every selectable strategy participates in a shared candidate-generation and reranking pipeline.

### 2.1 Profile signals

The serving layer accepts:

- Explicit ratings from 0.5 to 5.0.
- Watched movie IDs.
- Explicitly disliked/not-interested movie IDs.

Disliked movies are handled in two ways:

1. They are always excluded from results.
2. They are inserted into the preference profile as an implicit 1.0-star signal when no explicit rating exists.

This allows content and latent models to move away from disliked material rather than only hiding that exact movie.

### 2.2 Global candidate exclusions

Every model excludes:

- Already watched movies
- Explicitly disliked movies
- Movies with fewer than five ratings
- Movies explicitly rated in the active profile

The five-rating floor prevents extremely sparse movies from receiving unstable, overconfident scores.

### 2.3 Candidate pool size

For a request for `n` results, the primary model generates:

`min(240, max(n * 10, 60))`

For a typical top-10 request, this means a 100-item primary pool.

When the selected model produces candidates, only that model's candidates enter the final reranker. This keeps each For You tab honest and prevents supplemental content/popularity pools from collapsing every tab into the same shelf.

If the selected model cannot produce candidates after the user has supplied signal, the system falls back through:

- Content candidates, when ratings/preferences exist (`max(n * 4, 24)`)
- Popularity/genre candidates (`max(n * 3, 20)`)

The larger primary pool allows reranking to improve quality and diversity while preserving the selected model's identity.

### 2.4 Cold-start behavior

If the user has no watched or disliked movies, only the Popular tab returns Bayesian popularity candidates. Personalized and neural tabs return no picks until the user supplies taste signal. This avoids pretending to personalize without evidence.

As soon as the user supplies ratings or watched history, the selected personalized candidate generator is activated and owns its tab when it can produce candidates.

### 2.5 Shared score normalization

Candidate scores from different models are not directly comparable. For example, a predicted 4.3-star score and a cosine similarity of 0.61 are on different scales.

Therefore, scores are min-max normalized independently within each source model.

Source multipliers are then applied for the rare fallback pool that mixes sources:

- Selected primary model: 1.00
- Supplemental content model: 0.94
- Other supplemental source, normally popularity: 0.88

Duplicate movies are collapsed, retaining the strongest normalized source score.

### 2.6 Final reranking formula

For each unique candidate, the base score is:

`base = 0.85 * personalized_score + 0.15 * Bayesian_quality + 0.00 * novelty`

The weights are normalized before use, and at least one weight must be positive.

The zero novelty weight is intentional: it was selected by validation, not assumed in advance. Novelty is still measured and reported. Diversity remains active through the next step.

### 2.7 Diversity-aware selection

The final list is selected greedily using an MMR-like penalty:

`selection_score = base_score - 0.08 * maximum_genre_similarity_to_selected_items`

Genre similarity is Jaccard overlap. This discourages the final list from containing ten nearly identical genre combinations while preserving the stronger personalization-first weights selected during tuning.

## 3. Model 1: Bayesian popularity

### 3.1 Purpose

Popularity is the cold-start baseline and emergency fallback. It requires no user profile and remains available if a personalized model has no usable signal.

### 3.2 Training calculation

For each movie, training aggregates:

- Rating count `v`
- Mean rating `R`
- Global rating mean `C`

The minimum-count prior `m` is the 60th percentile of movie rating counts.

The weighted score is:

`weighted_score = (v / (v + m)) * R + (m / (v + m)) * C`

This shrinks movies with very few ratings toward the global mean. Two 5-star ratings therefore cannot outrank a well-established movie solely because their raw mean is 5.0.

### 3.3 Personalized popularity mode

When a profile exists, `because_you_watched`:

- Collects genres from movies rated at least 4.0.
- Counts genre overlap for every candidate.
- Requires at least one shared genre.
- Scores candidates as `genre_overlap * weighted_score`.

### 3.4 Artifact

- `models/popularity.csv`

### 3.5 Strengths and limitations

Strengths:

- Reliable for new users.
- Fast and interpretable.
- Stable against low-count rating noise.
- Provides a safe fallback when personalized artifacts fail.

Limitations:

- Weak personalization.
- Reinforces popularity bias.
- Cannot discover niche taste without another model.

## 4. Model 2: TF-IDF content matching

### 4.1 Purpose

Content matching supports:

- `More like this` recommendations.
- Early personalization with only a handful of ratings.
- Supplemental candidate generation for other models.
- Movement away from explicitly disliked content.

### 4.2 Vectorizer configuration

The final TF-IDF vectorizer uses:

| Parameter | Value |
| --- | --- |
| Minimum document frequency | 2 |
| Maximum document frequency | 0.60 |
| Maximum features | 120,000 |
| N-gram range | 1 to 2 |
| Sublinear term frequency | Enabled |

Unigrams capture individual genres/tags, while bigrams capture useful local combinations. Sublinear term frequency prevents heavily repeated tags from dominating linearly.

### 4.3 Profile construction

Known ratings are retained when they are at least 0.5 away from the neutral anchor of 3.0.

Each retained movie receives weight:

`weight = rating - 3.0`

Therefore:

- Ratings above 3.0 contribute positive content signals.
- Ratings below 3.0 contribute negative content signals.
- Neutral ratings do not distort the profile.

Weights are normalized by their total absolute magnitude. The weighted sparse movie vectors are combined into one profile vector. Cosine similarity between that profile and every movie document becomes the raw content score.

### 4.4 Efficient ranking

Only positive-similarity, non-excluded candidates are eligible. `numpy.argpartition` selects a top candidate pool without fully sorting the entire 87,585-movie catalog.

### 4.5 Artifacts

- `models/content_tfidf_vectorizer.joblib`
- `models/content_tfidf_matrix.npz`
- `models/content_movie_ids.npy`

### 4.6 Strengths and limitations

Strengths:

- Works with limited user history.
- Handles signed likes and dislikes.
- Provides explainable similarity.
- Does not depend on the target user appearing in MovieLens.

Limitations:

- Current metadata is mostly genres and tags.
- Cannot directly learn cross-genre community taste patterns.
- Recommendations can become too similar without shared reranking.

## 5. Model 3: Regularized collaborative matrix factorization

### 5.1 Purpose

This is the primary personalized model and the strongest final model on rating prediction and most ranking metrics.

### 5.2 Sparse user-item matrix

Training creates a CSR sparse matrix with:

- Rows: MovieLens users
- Columns: movies
- Values: explicit ratings

User and movie IDs are sorted and mapped to matrix positions. The sparse representation avoids materializing a dense user-by-movie matrix.

### 5.3 Bias removal

Let:

- `C` be the global mean rating.
- `R_i` be movie `i`'s mean rating.
- `n_i` be movie `i`'s rating count.
- `25` be the selected movie-bias shrinkage strength.

The regularized movie bias is:

`movie_bias_i = (R_i - C) * n_i / (n_i + 25)`

For every user row, the system computes that user's observed mean and subtracts both:

- The user mean
- The regularized movie bias for each observed item

The factorization therefore models residual preference rather than simply learning that highly rated movies are good for everyone.

### 5.4 Truncated SVD

Final settings:

| Parameter | Value |
| --- | ---: |
| Latent dimensions | 64 |
| Power iterations | 7 |
| Random seed | 42 |
| Movie-bias shrinkage | 25.0 |

`TruncatedSVD` produces:

- A user-factor matrix
- A 64-dimensional item-factor matrix
- A fitted SVD object

### 5.5 New-user fold-in

FilmFlicker users are not MovieLens training users. The production system therefore does not look up a stored MovieLens user vector. It solves a regularized least-squares fold-in problem from the new profile.

First, the profile baseline is shrunk toward the global mean:

`baseline = (sum(profile ratings) + 3 * global_mean) / (profile_count + 3)`

For rated items with factor matrix `A`, the centered rating vector removes this baseline and the corresponding movie biases.

The new user vector is:

`user_vector = solve(A.T * A + 0.15 * I, A.T * centered_ratings)`

The final predicted rating is:

`prediction = baseline + item_factor dot user_vector + movie_bias`

Predictions are clipped to the valid 0.5 to 5.0 rating range.

### 5.6 Final selected configuration

| Parameter | Before tuning | Final winner |
| --- | ---: | ---: |
| Latent dimensions | 32 | 64 |
| Fold-in regularization | 0.35 | 0.15 |
| Profile baseline strength | 5.0 | 3.0 |
| Movie-bias strength | 25.0 | 25.0 |

The final model is more expressive and more responsive to a small FilmFlicker profile, while still retaining regularized user and movie baselines.

### 5.7 Artifact

- `models/collaborative_svd.joblib`

The serving artifact stores only item factors, movie IDs and their index, movie biases, the global mean, and the exact fold-in configuration. Training-only SVD state, MovieLens user factors, user IDs, and user means are intentionally omitted. This reduced the artifact from about 93 MiB to about 22 MiB and avoids loading data the API can never use.

## 6. Model 4: KMeans taste neighborhoods

### 6.1 Purpose

Clustering is primarily an exploration and insight model. It supports the Taste Lab by describing meaningful groups such as `Drama + Romance` instead of exposing arbitrary labels such as `Cluster 2`.

### 6.2 Movie features

Each movie receives:

- Multi-hot genre features
- `log(1 + rating_count)`
- Mean rating

All features are standardized using `StandardScaler`.

### 6.3 KMeans configuration

| Parameter | Value |
| --- | ---: |
| Number of clusters | 15 |
| Initialization runs | 10 |
| Random seed | 42 |

### 6.4 Human-readable cluster profiles

For every learned cluster, training stores:

- Two most frequent meaningful genres
- Median release year
- Movie count
- A label built from the top genres

`IMAX` and `(no genres listed)` are excluded from label generation.

### 6.5 User-cluster affinity

Ratings at or below 2.5 do not add positive affinity. For every rating above 2.5:

`cluster_affinity += rating - 2.5`

Recommendations may come from the user's three strongest clusters. Within those clusters, the score favors cluster affinity first, then Bayesian quality, then moderate novelty. This keeps Taste neighborhoods exploratory instead of duplicating the Popular tab.

### 6.6 Artifact

- `models/clustering.joblib`

### 6.7 Intended role

The final metrics show that clustering is weaker than collaborative and content matching as a top-10 ranker. It should remain deployed for exploration, explanations, and analytics rather than serving as the default recommendation path.

## 7. Model 5: Two-tower taste embeddings

### 7.1 Model identity

The deployed taste-embedding artifact is a genuine two-tower neural recommender trained directly from ratings. It is not derived from collaborative SVD factors.

The UI and API call it `Taste embeddings` because serving folds new app profiles into a learned taste vector space. Internally the artifact declares `training_mode=two_tower_neural`, and `scripts/validate_model_export.py` fails stale `svd_embedding_fallback` artifacts.

The trainer is implemented in NumPy so local training and deployment do not require TensorFlow.

### 7.2 Training architecture

The offline trainer learns:

- User embeddings
- Movie embeddings
- User and movie biases
- A user tower: dense projection from user embedding to a shared latent space
- An item tower: dense projection from movie embedding plus genre features to the same latent space

The model is trained on rating residuals:

`rating - global_mean`

The prediction during training is:

`dot(user_tower(user_embedding), item_tower(movie_embedding, genres)) + user_bias + movie_bias`

Only the item side is exported for serving because FilmFlicker users are new profiles, not MovieLens users with learned user IDs.

### 7.3 New-profile vector

At serving time, a new user's rated movies are passed through the learned item tower. Known item vectors are weighted by rating deviation from the regularized profile baseline and averaged into an implied user vector.

Predictions combine:

- Profile baseline
- Dot product between the implied user vector and learned item-tower output
- Movie bias

### 7.4 Artifact

- `models/neural_weights.npz`

The historical artifact filename remains `neural_weights.npz` for compatibility. The artifact contains serving weights plus provenance fields such as `training_mode`, `embedding_dim`, `latent_dim`, `epochs`, and `validation_rmse_residual`.

### 7.5 Intended role

The taste embedding model is deployable as a ranking challenger. In the August 2 full-32M release-gate evaluation it overlaps collaborative on only 17.3% of top-10 recommendations, proving it contributes different candidate signal. Collaborative remains stronger for rating prediction, while content-based matching is stronger for top-10 ranking.

## 8. Evaluation design

### 8.1 Why the protocol changed

The earliest project evaluation used a random row-level 80/20 split. That can allow a model to train on interactions occurring after a test interaction and does not faithfully represent the real task: predict what a user will like next from only their earlier history.

The final protocol uses chronological per-user holdout. It also samples complete user histories rather than isolated rating rows.

### 8.2 Complete-history sampling

For computationally bounded evaluation:

- Users are shuffled with random seed 42.
- Entire user histories are selected until the requested rating budget is reached.
- No selected user has only a random subset of their history retained.

The final evaluation uses all 32,000,204 ratings from MovieLens 32M.

Production artifacts are trained separately on all 32,000,204 ratings. The neural production artifact also uses all 32,000,204 ratings instead of the earlier one-million-rating sample.

### 8.3 Outer chronological test split

For each eligible user:

- Ratings are sorted by `userId` and `timestamp`.
- Up to the latest five interactions are held out.
- At least the first five interactions remain available for training in final evaluation.
- The test fraction target is 20% per user.

Final split counts:

| Split | Ratings |
| --- | ---: |
| Pre-test training | 31,000,868 |
| Untouched test interactions | 999,336 |

### 8.4 Inner validation split used for tuning

The outer test interactions were not scored during parameter selection.

The earlier outer-training data was split again:

| Tuning split | Ratings |
| --- | ---: |
| Model fit | 949,194 |
| Validation | 19,065 |
| Reserved outer test | 31,615 |

The tuning search evaluated 200 fixed validation users. Eligible users had:

- At least five fit interactions
- At least one validation movie rated 4.0 or above

### 8.5 Relevance definition

A held-out movie is considered relevant when its rating is at least 4.0.

This is a strict observed-positive evaluation:

- A held-out liked movie is relevant.
- An unobserved movie is treated as unknown, not a confirmed dislike.
- Recommending an excellent but unobserved movie receives no offline credit.

That property makes top-10 percentages appear low in a large catalog and is one reason cross-project comparisons require identical protocols.

### 8.6 Models retrained inside evaluation

To avoid leakage, evaluation retrains rating-sensitive artifacts on the pre-test data:

- Collaborative SVD
- KMeans clustering
- Independent two-tower neural taste embeddings
- Popularity counts and Bayesian scores

The content model is rating-independent and uses movie metadata/tags, so its production content artifact is loaded directly. Each test user's profile still contains only pre-test ratings.

### 8.7 Ranking user sample

The final ranking report uses 1,000 randomly sampled eligible users with seed 42.

Rating-prediction RMSE and MAE use a 100,000-row sample from the 999,336 held-out rating rows.

## 9. Metric definitions

### 9.1 RMSE

Root Mean Squared Error measures predicted-rating error:

`RMSE = sqrt(mean((prediction - actual)^2))`

Lower is better. Larger mistakes are penalized more heavily because errors are squared.

Only collaborative and taste embeddings directly predict ratings, so only those models receive RMSE and MAE.

### 9.2 MAE

Mean Absolute Error is:

`MAE = mean(abs(prediction - actual))`

Lower is better. It is easier to interpret than RMSE because it represents average absolute star-rating error.

### 9.3 Precision@10

`Precision@10 = relevant recommended movies / returned top-10 movies`

If Precision@10 is 1.40%, approximately 14 of every 1,000 recommendation slots match an observed future liked movie under this protocol.

This does not mean the other 986 recommendations are necessarily bad; most are unobserved and therefore unknown.

### 9.4 Recall@10

`Recall@10 = relevant recommended movies / all held-out relevant movies`

Recall measures how much of the known future-like set was recovered by ten recommendation slots.

### 9.5 Hit Rate@10

Hit Rate is 1 for a user when at least one top-10 recommendation appears in that user's held-out liked set, otherwise 0. The report averages this over users.

A 12.50% Hit Rate@10 means 125 of 1,000 evaluated users received at least one observed future liked movie in their first ten recommendations.

### 9.6 NDCG@10

Normalized Discounted Cumulative Gain rewards relevant movies more when they appear near the top of the list. It therefore captures both retrieval and ordering quality.

Binary relevance is used here: held-out ratings of at least 4.0 are relevant.

### 9.7 MRR@10

Mean Reciprocal Rank is the reciprocal rank of the first relevant movie:

- First position: 1.0
- Second position: 0.5
- Tenth position: 0.1
- No hit: 0.0

The value is averaged over users.

### 9.8 Intra-list diversity

For every pair of recommended movies, genre distance is:

`1 - genre_Jaccard_similarity`

The metric averages pairwise distance across each list and then across users. Higher values indicate less repetitive genre composition.

### 9.9 Mean novelty bits

Novelty uses self-information based on movie interaction frequency:

`novelty(movie) = -log2((rating_count + 1) / (total_interactions + catalog_size))`

Higher values indicate less frequently rated movies.

### 9.10 Catalog coverage

`coverage = unique recommended movie IDs / 87,585 catalog movies`

The denominator includes the entire processed catalog, including movies that may be excluded by the five-rating minimum. Coverage is therefore intentionally conservative.

## 10. Development evaluation chronology

### 10.1 Historical random-split baseline

The earliest recorded evaluation used a random 80/20 row split. Historical results included:

| Metric | Collaborative | Taste embeddings |
| --- | ---: | ---: |
| RMSE | 1.4867 | 0.9367 |
| MAE | 1.0892 | 0.7162 |

Historical Precision@10:

| Model | Precision@10 |
| --- | ---: |
| Popularity | 0.04% |
| Content | 0.36% |
| Collaborative | 0.32% |
| Clustering | 0.28% |
| Taste embeddings | 0.32% |

These results are retained as development history only. They must not be compared directly with the final metrics because the data split, recommender logic, candidate handling, and user sampling all changed.

### 10.2 First chronological smoke test

A 30-user chronological smoke test verified the redesigned evaluation and serving logic before larger runs.

Selected results:

| Metric | Collaborative | Taste embeddings |
| --- | ---: | ---: |
| RMSE | 0.9166 | 0.9312 |
| MAE | 0.6837 | 0.6931 |
| Precision@10 | 1.33% | 1.00% |
| Hit Rate@10 | 10.00% | 10.00% |
| Diversity | 81.3% | 82.6% |

Because this cohort contained only 30 users, it was used to catch implementation failures rather than to support a quality claim.

### 10.3 Pre-tuning 250-user temporal evaluation

Before formal parameter tuning, the temporal evaluator was run on 250 users.

Rating prediction:

| Model | RMSE | MAE |
| --- | ---: | ---: |
| Collaborative | 0.9089 | 0.6820 |
| Taste embeddings | 0.9293 | 0.6974 |

Ranking:

| Model | P@10 | R@10 | Hit@10 | NDCG@10 | MRR@10 | Diversity | Novelty | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | 1.56% | 4.23% | 13.2% | 2.76% | 3.96% | 71.28% | 12.73 | 0.44% |
| Content | 1.76% | 4.69% | 14.8% | 3.67% | 5.94% | 71.34% | 12.62 | 0.58% |
| Collaborative | 1.96% | 5.60% | 17.6% | 4.06% | 6.39% | 81.58% | 11.70 | 0.34% |
| Clustering | 1.24% | 3.45% | 10.4% | 2.78% | 4.64% | 67.29% | 12.99 | 0.33% |
| Taste embeddings | 1.64% | 4.37% | 13.6% | 3.23% | 5.16% | 81.20% | 11.77 | 0.34% |

This 250-user cohort is not directly comparable to the final 1,000-user cohort because changing the sample size changes the seeded user sample. It was used as an intermediate quality check.

## 11. Hyperparameter tuning

### 11.1 Tuning objective

The validation objective was:

`0.40 * NDCG@10 + 0.30 * HitRate@10 + 0.20 * Recall@10 + 0.10 * catalog_coverage`

If diversity fell below 0.75, the score received this penalty:

`0.05 * (0.75 - diversity)`

This prioritizes ranking quality and the chance of giving a user at least one useful result, while preventing a configuration from winning solely by recommending a narrow repeated list.

### 11.2 Round 1: latent dimensions

All other parameters remained at the then-current values.

| Candidate | P@10 | R@10 | Hit@10 | NDCG@10 | MRR@10 | Diversity | Coverage | Objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 factors | 0.90% | 4.58% | 9.0% | 3.07% | 3.69% | 80.52% | 0.29% | 0.04874 |
| 32 factors | 1.00% | 4.92% | 9.0% | 3.31% | 3.72% | 80.59% | 0.32% | 0.05040 |
| 64 factors | 0.95% | 4.50% | 9.5% | 3.42% | 4.57% | 80.71% | 0.34% | 0.05151 |

Winner: 64 factors.

The 64-factor model had the best NDCG, Hit Rate, MRR, diversity, coverage, and combined objective, despite slightly lower precision/recall than 32 factors on this validation sample.

### 11.3 Round 2: new-user regularization

The selected 64-factor artifact was evaluated with four serving-time fold-in/profile configurations.

| Candidate | Fold-in reg. | Baseline strength | P@10 | R@10 | Hit@10 | NDCG@10 | MRR@10 | Diversity | Coverage | Objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Responsive | 0.15 | 3.0 | 1.00% | 4.67% | 10.0% | 3.89% | 5.57% | 80.51% | 0.38% | 0.05525 |
| Current | 0.35 | 5.0 | 0.95% | 4.50% | 9.5% | 3.42% | 4.57% | 80.71% | 0.34% | 0.05151 |
| Stable | 0.75 | 5.0 | 0.90% | 4.67% | 9.0% | 3.38% | 4.36% | 80.37% | 0.30% | 0.05016 |
| Conservative | 0.50 | 10.0 | 0.95% | 4.83% | 9.5% | 3.43% | 4.44% | 80.50% | 0.33% | 0.05223 |

Winner: responsive fold-in with regularization 0.15 and baseline strength 3.0.

The winner improved NDCG, Hit Rate, MRR, and coverage, indicating that FilmFlicker's small new-user profiles benefited from less aggressive shrinkage.

### 11.4 Round 3: reranking weights

The 64-factor responsive collaborative model was combined with four reranking configurations.

| Candidate | Personal | Quality | Novelty | Diversity penalty | P@10 | R@10 | Hit@10 | NDCG@10 | MRR@10 | Diversity | Coverage | Objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 0.72 | 0.23 | 0.05 | 0.16 | 1.00% | 4.67% | 10.0% | 3.89% | 5.57% | 80.51% | 0.38% | 0.05525 |
| Personalized | 0.82 | 0.13 | 0.05 | 0.12 | 1.05% | 5.17% | 10.5% | 4.04% | 5.60% | 78.84% | 0.42% | 0.05842 |
| Accuracy-first | 0.85 | 0.15 | 0.00 | 0.08 | 1.05% | 5.17% | 10.5% | 4.06% | 5.69% | 78.42% | 0.40% | 0.05848 |
| Discovery | 0.68 | 0.20 | 0.12 | 0.18 | 1.05% | 5.25% | 10.0% | 3.50% | 4.63% | 80.78% | 0.40% | 0.05488 |

Winner: accuracy-first.

It narrowly beat the personalized candidate on NDCG, MRR, and the combined objective while remaining above the 0.75 diversity floor.

### 11.5 Validation improvement summary

The original 32-factor/current configuration had objective 0.05040. The final selected configuration reached 0.05848.

Relative validation objective improvement:

`(0.05848 - 0.05040) / 0.05040 = approximately 16.0%`

This is a validation improvement, not a claim of 16% commercial engagement improvement.

### 11.6 Tuning runtime

The complete three-round search took approximately 300.5 seconds in the recorded development environment.

## 12. Final untouched evaluation

The final report was generated at `2026-08-02T03:41:16Z`.

Evaluation runtime was approximately 865.4 seconds. This run includes an extra paired pass for the previous collaborative configuration and retrains the independent two-tower neural evaluator on the full pre-test split.

### 12.1 Final rating-prediction metrics

| Model | RMSE | MAE |
| --- | ---: | ---: |
| Collaborative hybrid | 0.8993 | 0.6716 |
| Taste embeddings | 0.9656 | 0.7453 |

Interpretation:

- Collaborative has the lower error on both metrics.
- The average collaborative absolute rating error is approximately 0.67 stars.
- Taste embeddings are independently trained and useful as a distinct challenger, but they do not beat collaborative as a raw rating predictor.

### 12.2 Final top-10 ranking metrics

| Model | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.17% | 0.44% | 1.7% | 0.43% | 0.87% |
| Content-based | **1.50%** | **4.30%** | **12.5%** | **3.28%** | **5.19%** |
| Collaborative hybrid | 1.00% | 2.83% | 9.0% | 2.10% | 3.36% |
| Taste neighborhoods | 0.00% | 0.00% | 0.0% | 0.00% | 0.00% |
| Taste embeddings | 0.63% | 1.77% | 5.6% | 1.23% | 1.66% |

Content-based matching leads Precision, Recall, Hit Rate, NDCG, and MRR on the 1,000-user full-32M gate. That makes it the best offline top-10 recommendation model under the current protocol. Collaborative remains better for rating prediction and diversity, but it should no longer be described as the strongest top-10 ranker.

### 12.3 Uncertainty and paired baseline comparison

The August 2 committed evaluation records normal-approximation 95% confidence intervals over 1,000 per-user ranking outcomes. For the content-based serving strategy:

| Metric | Point estimate | 95% interval |
| --- | ---: | ---: |
| Precision@10 | 1.50% | 1.23% to 1.77% |
| Recall@10 | 4.30% | 3.50% to 5.10% |
| Hit Rate@10 | 12.50% | 10.45% to 14.55% |
| NDCG@10 | 3.28% | 2.64% to 3.91% |

The previous 32-factor configuration was retrained on the same train split and scored for the same users. The tuned-minus-previous paired deltas were:

| Metric | Mean delta | 95% interval |
| --- | ---: | ---: |
| Precision@10 | +0.30 percentage points | +0.12 to +0.48 points |
| Recall@10 | +0.58 percentage points | -0.13 to +1.28 points |
| Hit Rate@10 | +2.80 percentage points | +1.17 to +4.43 points |
| NDCG@10 | +0.58 percentage points | +0.03 to +1.13 points |

The tuned collaborative configuration improves Precision@10, Hit Rate@10, and NDCG@10 over the previous collaborative configuration with intervals above zero; Recall@10 is directionally higher but its interval includes zero. It also has lower rating error (RMSE 0.8993 versus 0.9043).

The neural taste model is also compared pairwise with collaborative:

| Metric | Neural minus collaborative | 95% interval |
| --- | ---: | ---: |
| Precision@10 | -0.37 percentage points | -0.60 to -0.14 points |
| Recall@10 | -1.06 percentage points | -1.78 to -0.33 points |
| Hit Rate@10 | -3.40 percentage points | -5.41 to -1.39 points |
| NDCG@10 | -0.87 percentage points | -1.44 to -0.30 points |
| MRR@10 | -1.71 percentage points | -2.64 to -0.78 points |

The neural intervals are below zero against collaborative, so the full-32M gate does show weaker neural top-10 ranking. Taste embeddings remain meaningful because they are independently trained and have different top-10 output, but they are not the best recommender under the current offline protocol.

### 12.4 Final beyond-accuracy metrics

| Model | Diversity | Mean novelty bits | Catalog coverage |
| --- | ---: | ---: | ---: |
| Popularity | 56.70% | 15.91 | 0.24% |
| Content-based | 54.64% | 14.73 | **2.30%** |
| Collaborative hybrid | **86.56%** | 11.36 | 0.29% |
| Taste neighborhoods | 59.96% | **21.83** | 0.19% |
| Taste embeddings | 80.22% | 11.19 | 0.16% |

Interpretation:

- Content provides the strongest ranking metrics and reaches the largest unique portion of the catalog.
- Collaborative provides the strongest genre diversity and the best rating prediction.
- Taste neighborhoods recommend the least popular material on average but have the weakest ranking quality.
- Popularity is more diverse than its name might imply because the reranker applies a diversity pass to its final list.
- Absolute catalog coverage remains low and should be understood in the context of only 10,000 total recommendation slots and an 87,585-movie denominator.

## 13. Industry and research context

There is no universal industry threshold that makes an offline recommender production-ready. NDCG, Hit Rate, Precision, and Recall depend heavily on:

- Dataset
- Catalog size
- Candidate filtering
- Train/test strategy
- Number of held-out interactions
- Relevance threshold
- Whether all catalog items or sampled negatives are ranked
- User eligibility rules
- Metric aggregation

Research has shown that changing these choices can produce different or opposite model comparisons:

- Canamares, Castells, and Moffat, `Offline evaluation options for recommender systems`: https://doi.org/10.1007/s10791-020-09371-3

A 2025 MovieLens 1M baseline study reported approximately 0.8757 RMSE for SVD and 0.8745 for a hybrid:

- `Evaluating Recommender System using Baseline Approaches`: https://doi.org/10.1016/j.procs.2025.04.495

FilmFlicker's collaborative RMSE of 0.8993 is somewhat higher, but the figures are not directly comparable because FilmFlicker uses the full MovieLens 32M dataset and chronological per-user holdout rather than that study's exact data/protocol.

The serving architecture is consistent with the standard candidate-generation, scoring, and reranking pattern described by Google:

- https://developers.google.com/machine-learning/recommendation/overview/types

Commercial systems such as Netflix combine offline experimentation with live A/B tests focused on retention and engagement:

- Gomez-Uribe and Hunt, `The Netflix Recommender System`: https://doi.org/10.1145/2843948

Therefore, the correct claim is:

- FilmFlicker has a reproducibly evaluated, deployment-ready portfolio recommender.
- Its content-based model has the strongest top-10 ranking point estimates among the included serving strategies under the documented offline protocol.
- Its advantage over the previous collaborative configuration is not statistically conclusive in the paired 1,000-user test.
- It has not been validated for commercial engagement or retention because it has no meaningful live traffic.

## 14. Production artifact training

After validation selected the winning parameters, the constants were promoted into:

- `src/recommenders/collaborative.py`
- `src/recommenders/ranking.py`

The complete production training command was:

```bash
FILMFLICKER_DATA_DIR=data/ml-32m backend/.venv/bin/python scripts/train_models.py --neural-sample-size 32000204 --neural-epochs 8
```

Training order:

1. Load processed movies, ratings, and tags.
2. Aggregate and save the popularity table.
3. Fit and save the content TF-IDF vectorizer/matrix.
4. Fit and save the 64-factor collaborative SVD.
5. Fit and save the 15-cluster KMeans model and cluster profiles.
6. Train and export the two-tower neural taste-embedding artifact.

The August 2 neural production refresh trained the two-tower artifact on all 32,000,204 ratings in approximately 242.8 seconds on the development machine. The full-32M evaluation run completed in approximately 865.4 seconds. Runtime is hardware-dependent and should not be treated as a deployment SLA.

## 15. Artifact inventory

| Artifact | Purpose |
| --- | --- |
| `models/popularity.csv` | Bayesian movie quality/count table |
| `models/content_tfidf_vectorizer.joblib` | Fitted TF-IDF vocabulary and configuration |
| `models/content_tfidf_matrix.npz` | Sparse movie-content vectors |
| `models/content_movie_ids.npy` | TF-IDF row-to-movie mapping |
| `models/collaborative_svd.joblib` | Serving-only item factors, biases, ID maps, and fold-in settings |
| `models/clustering.joblib` | Scaler, KMeans model, labels, and readable cluster profiles |
| `models/neural_weights.npz` | NumPy-served two-tower neural taste embedding artifact |
| `models/tuning_results.json` | Validation search and selected configuration |
| `models/metrics.json` | Final untouched evaluation report |
| `models/artifact_manifest.json` | Release generation ID, catalog provenance, configuration, and SHA-256 checksums |

The deployed backend caches these artifacts once per process. It does not retrain models during requests.

## 16. Deployment verification

### 16.1 Automated tests

Final test result:

- 55 tests passed in the pinned backend environment.

Tests cover data utilities, evaluation metrics, temporal splitting, popularity, recommender integration, collaborative fold-in behavior, clustering behavior, profile baselines, reranking, model exclusions, JWT issuer trust, UUID subjects, poster redirect allowlisting, SQLite connection ownership, and API year validation.

### 16.2 Artifact validation

Command:

```bash
backend/.venv/bin/python scripts/validate_model_export.py
```

Result:

- Model export validation passed.

### 16.3 Compilation and frontend checks

The following passed:

- Backend/source/script Python compilation
- Frontend TypeScript production build
- Frontend ESLint
- Git whitespace/error check

### 16.4 Real-artifact serving sanity check

A warm profile containing ratings for several well-known movies and one explicit dislike was used against the real exported artifacts.

Verified:

- 12 results returned.
- All 12 were unique.
- No watched movie appeared.
- No explicitly disliked movie appeared.
- The loaded artifact contained 64 dimensions.
- Fold-in regularization was 0.15.
- Profile baseline strength was 3.0.

### 16.5 In-process latency check

Thirty warm-profile recommendation generations produced:

| Latency statistic | Result |
| --- | ---: |
| Median | 145.0 ms |
| p95 | 169.9 ms |

This measures in-process model recommendation time after artifact loading. It does not include:

- Network latency
- Authentication
- Database queries
- Poster API/proxy work
- Cold process startup
- Hosting platform scheduling

### 16.6 Dependency compatibility

Training and backend requirements pin:

`scikit-learn==1.6.1`

The committed sklearn artifacts were validated with scikit-learn 1.6.1. The deployment backend and training virtual environment both use the pinned 1.6.1 version, avoiding cross-version artifact loads in production.

## 17. Deployment roles and recommendation policy

### 17.1 Content-based

Status: strongest top-10 recommendation model and production `more like this` model.

Why:

- Best Precision@10, Recall@10, Hit Rate@10, NDCG@10, and MRR@10 in the full-32M gate.
- Highest catalog coverage.
- Works with short profiles.
- Uses negative preference signals.

### 17.2 Collaborative hybrid

Status: best rating-prediction model and strong diversity-focused personalized tab.

Why:

- Best RMSE and MAE.
- Strongest intra-list diversity at 86.56%.
- Personalized fold-in works for new FilmFlicker users.
- Still beats the previous collaborative configuration on the paired full-32M gate.

### 17.3 Popularity

Status: production cold-start and failure fallback.

Why:

- Does not require user history.
- Bayesian shrinkage is robust.
- Produces a complete list when personalized models have no signal.

### 17.4 Taste embeddings

Status: deployable ranking challenger model.

Why:

- Independently trained two-tower neural recommender.
- Different top-10 output from collaborative, with 17.3% mean top-10 overlap in the full-32M release-gate evaluation.
- Lower ranking and rating point estimates than collaborative and content-based on the 1,000-user gate, so it should not replace either as the primary recommender.

### 17.5 Taste neighborhoods

Status: production insight/exploration model, not default ranking model.

Why:

- Human-readable cluster profiles improve Taste Lab.
- Ranking metrics trail content, collaborative, popularity, and neural in the full-32M gate.
- Useful as a different explanatory view rather than an accuracy winner.

## 18. Known limitations and residual risks

### 18.1 No online evaluation

There are no meaningful real users, A/B tests, click-through measurements, saves, completed watches, or retention metrics. Offline performance cannot prove user satisfaction.

### 18.2 Incomplete relevance labels

MovieLens only tells us what a user rated. A recommendation that does not appear in held-out history is unknown, not necessarily wrong.

### 18.3 Catalog freshness

The catalog ends in 2023 and was not enriched with TMDB during this build. This is the most visible data limitation for a public demo.

### 18.4 Popularity/exposure bias

MovieLens ratings are not missing at random. Users are more likely to encounter and rate already popular movies. Training and evaluation inherit this bias.

### 18.5 Coverage concentration

Final catalog coverage is 2.30% for content-based and below 0.30% for every other model over the 1,000-user test. Some of this follows mathematically from only 10,000 available recommendation slots, but repeated popular candidates also contribute.

### 18.6 Evaluation sampling

The final offline benchmark now uses all 32,000,204 ratings, which is more faithful but slower to reproduce. Earlier one-million-rating gates remain useful as development history only.

### 18.7 No confidence intervals

The final report averages 1,000 users and stores normal-approximation confidence intervals for Precision@10, Recall@10, Hit Rate@10, and NDCG@10. It does not replace online product evaluation.

### 18.8 Taste embedding role

The final embedding model is independently trained and validated as `two_tower_neural` on all 32,000,204 ratings, but the 1,000-user full-32M release gate shows weaker ranking than content-based and collaborative. Treat it as a credible challenger and source of different candidates, not as the default or best-performing recommender.

### 18.9 Cluster ranking quality

Clustering is meaningful for insights but should not be marketed as the most accurate recommendation strategy.

## 19. Exact reproduction workflow

All commands assume the repository root.

### 19.1 Install dependencies

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
cd frontend && npm install && cd ..
```

### 19.2 Download MovieLens 32M

```bash
backend/.venv/bin/python scripts/download_movielens.py 32m
```

### 19.3 Build the processed catalog

Without TMDB enrichment:

```bash
backend/.venv/bin/python scripts/build_catalog.py \
  --movielens-dir data/ml-32m \
  --tmdb-current-pages 0
```

With valid TMDB credentials, set `TMDB_API_KEY` or `TMDB_BEARER_TOKEN` before building.

### 19.4 Tune existing model parameters

```bash
backend/.venv/bin/python scripts/tune_models.py \
  --max-ratings 1000000 \
  --n-validation-users 200 \
  --output models/tuning_results.json
```

Important: the tuner writes the winning configuration but does not automatically modify production constants. Review the result and promote the winner into `collaborative.py` and `ranking.py` before production training.

Current promoted winner:

```text
n_components = 64
fold_in_regularization = 0.15
profile_baseline_strength = 3.0
movie_bias_strength = 25.0
personalized_weight = 0.85
quality_weight = 0.15
novelty_weight = 0.00
diversity_strength = 0.08
```

### 19.5 Train production artifacts on all ratings

```bash
FILMFLICKER_DATA_DIR=data/ml-32m backend/.venv/bin/python scripts/train_models.py --neural-sample-size 32000204 --neural-epochs 8
```

### 19.6 Run final chronological evaluation

```bash
backend/.venv/bin/python scripts/evaluate_models.py \
  --max-ratings 0 \
  --max-rating-predictions 100000 \
  --n-eval-users 1000 \
  --neural-sample-size 32000204 \
  --neural-epochs 8 \
  --output models/metrics.json
```

This command retrains temporary evaluation artifacts on the pre-test split and deletes them afterward. It does not overwrite the production SVD/clustering/popularity artifacts with evaluation-only versions.

### 19.7 Validate artifacts

```bash
backend/.venv/bin/python scripts/validate_model_export.py
```

### 19.8 Run tests and builds

```bash
python3 -m pytest -q
backend/.venv/bin/python -m compileall -q backend src scripts
cd frontend
npm run lint
npm run build
```

### 19.9 Confirm machine-readable outputs

Review:

```bash
cat models/tuning_results.json
cat models/metrics.json
cat data/processed/catalog_manifest.json
```

## 20. Guidance for future changes

When modifying a model:

1. Do not tune against `models/metrics.json` final-test results.
2. Add the candidate to the validation search.
3. Keep complete user histories and chronological order.
4. Compare against the current winner using the same validation users.
5. Preserve the diversity floor unless there is an explicit product reason to change it.
6. Promote parameters only after validation improves.
7. Retrain all dependent production artifacts.
8. Run the final evaluation once.
9. Update this document and `README.md` only with the committed evaluation result and clearly label its sample size.
10. Keep old and new protocols clearly separated.

If meaningful user traffic is ever available, add online metrics before making further claims:

- Recommendation impression rate
- Card click-through rate
- Save/watch conversion
- Explicit dislike rate
- Dismissal rate
- Time to first useful recommendation
- Session return rate
- Diversity of consumed, not merely displayed, movies

## 21. Final statement

FilmFlicker's model system is complete for its intended portfolio deployment. It has:

- Multiple complementary candidate generators
- Signed positive and negative feedback
- A shared production reranker
- New-user collaborative fold-in
- Leakage-resistant temporal evaluation
- Separate validation and untouched test stages
- Reproducible hyperparameter search
- Production artifacts trained from the full catalog, with the neural tower trained on all 32,000,204 ratings
- Artifact and behavior validation
- Documented performance and limitations

The strongest defensible July 24, 2026 claim is:

> FilmFlicker is a production-style hybrid movie recommendation system trained on MovieLens 32M. It includes a genuine NumPy-trained two-tower neural taste-embedding recommender trained on all 32,000,204 ratings. Its 1,000-user full-32M chronological release gate shows that content-based matching is the strongest top-10 recommender at 1.50% Precision@10, 4.30% Recall@10, 12.5% Hit Rate@10, and 3.28% NDCG@10, while collaborative remains the best rating predictor at 0.899 RMSE and 0.672 MAE. The neural taste model is independently trained and different, with 17.3% mean top-10 overlap with collaborative, but it is a challenger rather than the primary model.

It should not be described as commercially validated without real-user online experiments.

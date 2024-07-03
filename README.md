### Netflix Movie Recommender 
## Project Overview: Building and Validating a Recommendation Model Using Netflix Prize Data
Project Description:

This project aims to develop and validate a simple recommendation system using the Netflix Prize dataset. 

### Notebooks: 
The workflow is structured across two main notebooks:

**Exploration.ipynb**
- **Objective:** Perform data exploration and preprocessing to understand the characteristics and distributions within the Netflix Prize dataset.
- **Activities:**
  * Develop an approach to handle large datasets effectively.
  * Analyze data distributions, such as movie ratings and movie popularity.
  * Sample data to create representative subsets for model development and validation.

**Modeling.ipynb**
- **Objective:** Implement and evaluate the model.
- **Activities:**
  * Select appropriate recommendation algorithms.
  * Tune model parameters using grid search.
  * Validate models using accuracy metrics such as RMSE and MAE and user-centric metrics like hit rate, diversity, or novelty.
  * Generate sample recommendations to assess the effectiveness of the selected model.

### Other elements: 

**data:** Samples drawn from the original data for model training. \
**models:** Serialized models and parameters. \
**scripts:** Utility functions and classes to facilitate the data processing and analysis. \
**test:** Basic tests for MovieSampler class. 

### Additional Notes: 

**Data Source:** The Netflix Prize dataset ([Kaggle](https://www.kaggle.com/datasets/evanschreiner/netflix-movie-ratings?select=Netflix_User_Ratings.csv)) \
**Tools:** Python, Numpy, Pandas, Seaborn, surprise
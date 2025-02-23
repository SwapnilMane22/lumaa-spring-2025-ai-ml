# Movie Recommendation System

---

## Dataset
This project uses the **Movies Dataset** from Kaggle: [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

## Direct Dataset Download Link

Here is the direct dataset download link: [Download the dataset from OneDrive](https://binghamton-my.sharepoint.com/:x:/g/personal/smane_binghamton_edu/EaNa1OLCjhFEiRYv2h5q9PwBIyG4aL7c1yoUf39ulP0Qjg?e=6BcfhP).

### **Loading the Dataset**
- Download the dataset from the above link.
- Extract the dataset and place `movies_metadata.csv` inside an `archive/` folder in the project directory.

## Setup
### **Requirements**
Ensure you have **Python 3.8+** installed.

### **Create a Virtual Environment**
```bash
python -m venv env  # Create virtual environment
source env/bin/activate  # Activate on macOS/Linux
env\Scripts\activate  # Activate on Windows
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Pre-download the SBERT Model**
To avoid downloading the model each time the code runs, you can pre-download the **SBERT model** by running the following script:

1. Create a script called `download_model.py`:

```python
# download_model.py
from sentence_transformers import SentenceTransformer

# Pre-download the SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
```

2. Run the script to download and cache the model:

```bash
python download_model.py
```

This will ensure the **all-MiniLM-L6-v2** model is cached on your machine, speeding up future runs of the movie recommendation system.

## Running the Code
You can run the movie recommendation system using the command line:

```bash
python recommend.py "A thrilling action movie with superhero"
```

Alternatively, open and run the Jupyter Notebook if available.

---

## Demo Video

Here is the demo video: [Watch the video on OneDrive](https://binghamton-my.sharepoint.com/:v:/g/personal/smane_binghamton_edu/EZ8we6mKHYtOtX4GVRGLfqMB6Lc_ZZoq9H_fYliSZTR33g?e=zGUiKV).

---

## Results
Example output for the query: `"A thrilling action movie with a strong female lead"`
```
Top 5 Recommended Movies using TF-IDF:

1. Title: Fast Track: No Limits
   Similarity Score: 19.78%

2. Title: Run
   Similarity Score: 14.02%

3. Title: Death of a Superhero
   Similarity Score: 13.98%

4. Title: The Fourth Phase
   Similarity Score: 12.98%

5. Title: Dark Haul
   Similarity Score: 12.06%
```

```
Top 5 Recommended Movies using SBERT:

1. Title: Hero at Large
   Similarity Score: 58.24%

2. Title: Superheroes
   Similarity Score: 55.46%

3. Title: Somebody's Hero
   Similarity Score: 54.88%

4. Title: The Batman Superman Movie: World's Finest
   Similarity Score: 53.84%

5. Title: Super
   Similarity Score: 53.41%
```
The **TF-IDF** method is based on text frequency, while **SBERT** provides context-aware similarity.



## Notes
- The system uses **TF-IDF vectorization** and **SBERT embeddings** for recommendations.
- Adjust the **number of recommendations** (`top_n`) in the code as needed.
- CUDA-enabled GPU is recommended for SBERT.

## Importance of Feature Engineering and SBERT

### Feature Engineering
In machine learning, **feature engineering** plays a crucial role in improving the model’s performance. In this project, we combine the **movie title** and **overview** to create a new feature called `description`. This combination helps capture both the essence of the title and the detailed context in the overview, allowing the recommendation system to make more accurate suggestions based on the movie’s content. Feature engineering can significantly impact how the model interprets and processes data, especially when working with textual information, as it determines the quality of the input fed into the recommendation system.

### SBERT Model
The **SBERT (Sentence-BERT)** model is a powerful tool for semantic textual similarity. Unlike traditional methods like **TF-IDF**, which rely on word frequencies, SBERT captures the deeper context and meaning of a sentence or description. SBERT generates embeddings that are sensitive to the context in which words appear, which enables the model to recommend movies that are semantically similar, even if they don’t share the exact same words. This method provides higher quality and more relevant recommendations, especially for complex queries.

## Future Improvements
While the current system offers good recommendations using **TF-IDF** and **SBERT**, there are several ways to further enhance its capabilities:

1. **Model Fine-tuning**: Fine-tune the **SBERT** model using domain-specific data to make the embeddings more relevant to movie recommendations, improving the accuracy of similarity measures.

2. **Advanced NLP Techniques**: Integrate additional natural language processing techniques like Named Entity Recognition (NER) to understand key entities (e.g., directors, actors, genres) and incorporate them into the recommendation process.

3. **Real-time Recommendations**: Implement a real-time recommendation feature by tracking user interactions and preferences, providing immediate movie suggestions based on their viewing history.

4. **Interactive User Interface**: Create a user-friendly web or mobile application to make the recommendation system accessible and interactive, providing a seamless movie discovery experience.

## Salary Expectation
**$2500 per month** for **40 hours** of work per week.

Enjoy your personalized movie recommendations! 🎬🍿
---

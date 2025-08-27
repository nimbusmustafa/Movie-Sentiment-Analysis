
# 🎬 Movie Review Sentiment Analysis  

This project implements a **Naive Bayes classifier** from scratch to perform **sentiment analysis** on the IMDB movie reviews dataset. The model classifies reviews as **Positive** or **Negative** after preprocessing the text.  

---

## 🚀 Features
- Loads and processes the **IMDB Dataset** (50,000 reviews).
- Text preprocessing:
  - Removes URLs, HTML tags, and non-alphabet characters.
  - Converts text to lowercase.
  - Removes stopwords.
  - Applies stemming using **Porter Stemmer**.
- Converts labels (`positive`, `negative`) into binary values (`1` = positive, `0` = negative).
- Builds a **frequency dictionary** for words with sentiment counts.
- Implements **Naive Bayes algorithm** from scratch:
  - Calculates log prior probabilities.
  - Estimates log likelihoods for words.
- Trains and evaluates on **train/test split**.
- Achieves **~89% accuracy** on the test set 🎉
- Supports **custom user input** for real-time sentiment prediction.  

---

## 📂 Dataset
We use the **IMDB Dataset of 50,000 Movie Reviews**:  
- The dataset comes as a zipped CSV file: `IMDB Dataset.csv.zip`.  
- Each entry has:
  - `review`: The text of the movie review.  
  - `sentiment`: `positive` or `negative`.  

---

## 🛠️ Installation & Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/nimbusmustafa/Movie-Sentiment-Analysis.git
cd Movie-Sentiment-Analysis
pip install -r requirements.txt
````

### Requirements

* Python 3.x
* pandas
* numpy
* scikit-learn
* nltk

Make sure to download **NLTK stopwords**:

```python
import nltk
nltk.download('stopwords')
```

---

## 📊 Model Training

The workflow is as follows:

1. **Data Preprocessing**

   * Clean and normalize text using regex + stemming + stopword removal.
2. **Feature Engineering**

   * Build a frequency dictionary of words for positive/negative reviews.
3. **Train-Test Split**

   * 85% training, 15% testing.
4. **Naive Bayes Training**

   * Compute log prior and log likelihoods.
5. **Evaluation**

   * Accuracy on train set: **88.57%**
   * Accuracy on test set: **89.03%**

---

## 🔮 Example Prediction

```python
movie_review = "Screenplay structure, editing, cinematography, grand settings, graphics of new technology can go on and on. Best wishes for success."
movie_review = preprocess_text(movie_review)
p = predict_naive_bayes(movie_review, logprior, loglikelihood)

if p > 0:
    print("Positive Sentiment ✅")
else:
    print("Negative Sentiment ❌")
```

**Output:**

```
Positive Sentiment ✅
```

---

## 📈 Results

* **Train Accuracy:** 88.57%
* **Test Accuracy:** 89.03%
* Shows good generalization and balanced performance.

---

## 📌 Future Improvements

* Use **TF-IDF features** for better weighting.
* Replace Naive Bayes with advanced models like **Logistic Regression, SVM, or Neural Networks**.
* Deploy via a **Flask API** or **Streamlit app** for interactive predictions.

---



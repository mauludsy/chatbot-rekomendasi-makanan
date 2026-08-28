# 🍜 Food Recommendation Chatbot

A food recommendation chatbot that provides Indonesian food suggestions based on user preferences and queries. The system uses **Content-Based Filtering** with **TF-IDF** for text feature extraction and **Cosine Similarity** to identify recipes that are most relevant to the user's input.

The chatbot interface is developed using the **Streamlit framework**, providing an interactive and user-friendly way for users to search for food recommendations.

---

## 📌 About the Project

**Food Recommendation Chatbot** is a Natural Language Processing (NLP) and recommendation system project designed to help users find suitable Indonesian food based on their preferences.

The system utilizes an **Indonesian Food Recipes Dataset** containing information about food recipes, including:

* Food titles
* Ingredients
* Food categories
* Other recipe-related information

The recommendation pipeline consists of data cleaning, text preprocessing, feature engineering, TF-IDF feature extraction, and Cosine Similarity.

For each user query, the system transforms the input into a TF-IDF representation and compares it with the recipe representations available in the dataset. The system then ranks the recipes based on their similarity scores and returns the **Top 5 most relevant food recommendations**.

The final recommendation system is integrated into a **Streamlit-based chatbot interface**.

---

## 🎯 Objectives

The main objectives of this project are:

* Develop a simple and interactive food recommendation system.
* Recommend Indonesian food based on user preferences.
* Apply text preprocessing to recipe information.
* Combine relevant recipe information into a unified text feature.
* Transform recipe text into numerical features using TF-IDF.
* Calculate similarity between user queries and recipes using Cosine Similarity.
* Rank recipes based on similarity scores.
* Display the most relevant food recommendations through an interactive Streamlit interface.

---

## 📊 Dataset

The dataset used in this project is:

**Indonesian Food Recipes Dataset**

The dataset contains Indonesian food recipes with information related to food names, ingredients, categories, and other recipe attributes.

The dataset is used as the knowledge base for the recommendation system.

### Relevant Features

For the recommendation process, several relevant columns are utilized, including:

| Feature         | Description                    |
| --------------- | ------------------------------ |
| **Title**       | Name of the food recipe        |
| **Ingredients** | Ingredients used in the recipe |
| **Category**    | Food category                  |

These features are processed and combined into a new text feature called **`Recipe Text`**.

### Dataset Usage

The original dataset is **not included in this repository**.

The dataset is used locally during development, while the repository contains the code and notebook required to reproduce the recommendation system.

---

## 🔄 Recommendation Pipeline

The overall workflow of the project is:

```text
Recipe Dataset
      ↓
Data Cleaning
      ↓
Text Preprocessing
      ↓
Feature Engineering
      ↓
Create Recipe Text
      ↓
TF-IDF Vectorization
      ↓
Recipe TF-IDF Matrix
      ↓
User Input
      ↓
User TF-IDF Vector
      ↓
Cosine Similarity
      ↓
Similarity Ranking
      ↓
Top 5 Recommendations
      ↓
Streamlit Chatbot Interface
```

---

## 🧹 Data Preprocessing

The recipe data is cleaned and prepared before being used by the recommendation system.

### Recipe Title Cleaning

The food title is processed to remove unnecessary characters and normalize the text.

The preprocessing includes:

* Converting text to lowercase.
* Removing specific unwanted text patterns.
* Removing special characters.
* Removing excessive whitespace.
* Handling empty cleaned titles.

The processed title is stored in:

```text
Title Cleaned
```

### Ingredient Cleaning

The ingredients column is also cleaned before feature extraction.

The preprocessing includes:

* Converting text to lowercase.
* Removing URLs.
* Removing underscores.
* Removing numbers.
* Removing special characters.
* Removing excessive whitespace.

The processed ingredients are stored in:

```text
Ingredients Cleaned
```

---

## 🧩 Feature Engineering

After the preprocessing stage, relevant recipe information is combined into a single text feature called:

```text
Recipe Text
```

The feature is created by combining:

```text
Title Cleaned
      +
Ingredients Cleaned
      +
Category
      ↓
Recipe Text
```

For example:

```text
Title:
Ayam Geprek

Ingredients:
ayam cabai bawang garam

Category:
Makanan Utama
```

The resulting `Recipe Text` becomes:

```text
ayam geprek ayam cabai bawang garam makanan utama
```

This combined text is then used as the input for TF-IDF feature extraction.

---

## 🧠 Feature Extraction — TF-IDF

The recommendation system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform recipe text into numerical feature representations.

TF-IDF assigns weights to words based on their importance within a recipe and across the collection of recipes.

The TF-IDF representation allows the system to compare textual similarities between:

```text
User Query
      ↓
TF-IDF Vector
      ↓
Recipe TF-IDF Matrix
```

The resulting TF-IDF matrix is then used for similarity calculation.

---

## 🔍 Similarity Calculation — Cosine Similarity

After transforming the recipe text and user input into TF-IDF vectors, the system calculates their similarity using **Cosine Similarity**.

The process works as follows:

```text
User Input
"I want spicy food using chicken"
          ↓
     TF-IDF Vector
          ↓
   Cosine Similarity
          ↓
Compare with all recipes
          ↓
Similarity Scores
```

A higher similarity score indicates that the recipe is more relevant to the user's query.

For example:

| Recipe         | Similarity |
| -------------- | ---------: |
| Ayam Geprek    |       0.91 |
| Ayam Balado    |       0.87 |
| Ayam Rica-Rica |       0.82 |
| Soto Ayam      |       0.68 |
| Nasi Goreng    |       0.31 |

---

## 🏆 Recommendation Ranking

The similarity scores are sorted from the highest to the lowest value.

The system then selects the **Top 5 recipes** with the highest similarity scores.

The recommendation output contains:

* Food title
* Food category
* Ingredients
* Similarity score

Example:

```text
Top 5 Food Recommendations

1. Ayam Geprek
   Similarity: 0.91

2. Ayam Balado
   Similarity: 0.87

3. Ayam Rica-Rica
   Similarity: 0.82

4. Soto Ayam
   Similarity: 0.68

5. Ayam Bakar
   Similarity: 0.64
```

---

## 🤖 Chatbot Interaction

The recommendation system is integrated into a chatbot-style interface.

Users can enter natural-language queries describing the type of food they want.

### Example User Input

```text
Saya ingin makanan yang menggunakan ayam dan rasanya pedas
```

The system processes the query and returns the most relevant recipes based on the similarity between the user input and the recipe dataset.

### Recommendation Flow

```text
User
 ↓
Enter Food Preference
 ↓
Text Processing
 ↓
TF-IDF Transformation
 ↓
Cosine Similarity
 ↓
Ranking
 ↓
Top 5 Food Recommendations
```

---

## 🖥️ Web Application Interface
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a8fed0c4-2dfa-40fd-b9d3-791761756f11" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e0dc1a38-9b86-4ed7-94bb-acf790ddc11c" />



The chatbot interface is developed using **Streamlit**.

The interface provides an interactive environment where users can:

* Enter food preferences.
* Submit food-related queries.
* Receive recommended Indonesian food.
* View recipe information.
* View similarity scores.

The Streamlit application acts as the front-end interface for the recommendation system.

---

## 🛠️ Technologies

| Category                      | Tools / Libraries         |
| ----------------------------- | ------------------------- |
| **Programming Language**      | Python                    |
| **Data Processing**           | Pandas, NumPy             |
| **Text Processing**           | Python `re`               |
| **Feature Extraction**        | TF-IDF                    |
| **Similarity Calculation**    | Cosine Similarity         |
| **Machine Learning Library**  | Scikit-learn              |
| **Web Application Framework** | Streamlit                 |
| **Development Environment**   | Jupyter Notebook, VS Code |
| **Version Control**           | Git, GitHub               |

---

## 📁 Project Structure

```text
chatbot-rekomendasi-makanan/
│
├── app.py
├── rekomendasi_makanan.ipynb
├── README.md
└── .gitignore
```


## ⭐ Project Highlights

* Developed an **Indonesian food recommendation chatbot**.
* Implemented a **Content-Based Filtering** recommendation approach.
* Applied text preprocessing to recipe titles and ingredients.
* Combined recipe title, ingredients, and category into a unified text feature.
* Used **TF-IDF** for text feature extraction.
* Used **Cosine Similarity** to measure similarity between user queries and recipes.
* Implemented **Top-5 recommendation ranking**.
* Developed an interactive chatbot interface using the **Streamlit framework**.
* Integrated NLP and recommendation system techniques into a functional web application.
* Implemented the project using Python and Scikit-learn.

---

## 🔮 Future Development

Several features can be added to improve the recommendation system in the future:

* Add food price or budget-based recommendations.
* Add dietary preferences such as vegetarian or high-protein food.
* Add cooking time preferences.
* Add regional food preferences.
* Improve natural language understanding.
* Add recommendation history.
* Add more Indonesian food datasets.
* Improve the chatbot interface and user experience.

---

## 👤 Author

**M. Maulud Syafrizal**

Fresh Graduate — S1 Informatika
Universitas Amikom Yogyakarta

**Interests:** Data Science, Machine Learning, Natural Language Processing, Data Analysis, and Recommendation Systems

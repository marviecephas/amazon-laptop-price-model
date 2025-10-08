# 💻 Amazon Laptop Price Scraper & Predictor

## 📌 Project Overview
This project automates the process of **scraping laptop product data** from Amazon and uses the collected dataset to analyze **factors influencing laptop prices**.  
By combining **web scraping, data cleaning, and machine learning**, the project aims to help with **price comparison, market research, and data-driven purchasing decisions**.

---

## 🎯 Learning Outcomes
- **Web Scraping:** Extract product details (brand, RAM, storage, processor, price) using Python libraries like `requests`, `BeautifulSoup`, and optionally `Selenium` for dynamic pages.
- **Data Cleaning:** Convert unstructured text (e.g., "16GB RAM") into structured formats (e.g., 16).
- **Exploratory Data Analysis (EDA):** Use `pandas`, `matplotlib`, and `seaborn` to visualize pricing patterns.
- **Machine Learning:** Train regression models (`Linear Regression`, `Random Forest`, `XGBoost`) to predict laptop prices.
- **Feature Importance:** Identify which laptop specs (RAM, CPU, storage, GPU) most strongly impact pricing.
- **Price Tracking (Optional):** Scrape data over time to monitor price trends and forecast future drops.

---

## 🛠️ Tech Stack
- **Python**
- **Libraries:** `requests`, `beautifulsoup4`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Optional:** `Selenium`, `XGBoost`

---

## 📂 Project Workflow
1. **Set Up Environment**
   ```bash
   pip install requests beautifulsoup4 pandas numpy matplotlib seaborn scikit-learn


2. **Web Scraping**

   * Define target Amazon laptop product URLs.
   * Extract:

     * Laptop Name
     * Brand
     * Processor (e.g., i5, i7, Ryzen 5)
     * RAM (GB)
     * Storage (GB / TB)
     * Price ($)
     * Ratings / Reviews

3. **Data Cleaning**

   * Remove missing values.
   * Convert text → numeric (e.g., "512GB SSD" → 512).
   * Save structured data:

   ```python
   df.to_csv("laptops.csv", index=False)
   ```

4. **Exploratory Data Analysis**

   * Plot price distributions.
   * Correlation heatmap (features vs price).
   * Boxplots (RAM vs Price, CPU vs Price).

5. **Modeling**

   * Train models:

   ```python
   from sklearn.linear_model import LinearRegression
   from sklearn.ensemble import RandomForestRegressor
   ```

   * Evaluate using R², MSE.
   * Compare models to see which predicts price best.

6. **Feature Importance**

   * Use:

   ```python
   model.feature_importances_
   ```

   * Or **Permutation Importance**:

   ```python
   from sklearn.inspection import permutation_importance
   ```

7. **Price Tracking (Optional)**

   * Schedule scraping weekly.
   * Save dated CSVs (`laptops_2025-10-02.csv`).
   * Compare historical prices, visualize trends.

---

## 📊 Example Insights

* **RAM & CPU type** are the strongest predictors of price.
* Laptops with SSD storage are priced significantly higher than HDD-only models.
* Price drops can be tracked over weeks/months for better purchasing decisions.

---

## 🚀 Future Work

* Extend scraping to multiple e-commerce platforms.
* Deploy a **Flask/Django web app** where users can input laptop specs and get an estimated fair price.
* Build a **time-series forecasting model** for price prediction.

---

## 📌 Disclaimer

This project is for **educational purposes only**. Always check Amazon’s terms of service before scraping, and avoid sending too many requests that could overload their servers.





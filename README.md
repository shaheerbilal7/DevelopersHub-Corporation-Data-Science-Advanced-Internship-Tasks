# DevelopersHub-DataScience-Advanced-Internship-Tasks
Data science advanced internship tasks

# Task 1: Term Deposit Subscription Prediction (Bank Marketing)
Objective:
Predict whether a bank customer will subscribe to a term deposit as a result of a marketing
campaign.

Dataset:
Bank Marketing Dataset

Instructions:
* Load and explore the dataset
* Encode all categorical features properly
* Train classification models (e.g., Logistic Regression, Random Forest)
* Evaluate the models using Confusion Matrix, F1-Score, and ROC Curve

Results:
Here's a breakdown of what each visualization means:

1. **Logistic Regression:**

Visualization: A Logistic Regression Confusion Matrix.

What it shows: This matrix evaluates the performance of the Logistic Regression model on a classification task.

True Negatives (top-left): 974 instances were correctly predicted as class 0.

False Positives (top-right): 192 instances were incorrectly predicted as class 1 when they were actually class 0.

False Negatives (bottom-left): 230 instances were incorrectly predicted as class 0 when they were actually class 1.

True Positives (bottom-right): 837 instances were correctly predicted as class 1.

Metric: The F1-score for this model is reported as 79.864%. The F1-score is the harmonic mean of precision and recall, and it provides a single metric that balances both.

2. **Random Forest:**

Visualization: A Random Forest Confusion Matrix.

What it shows: This matrix performs the same evaluation for the Random Forest model.

True Negatives: 955

False Positives: 211

False Negatives: 146

True Positives: 921

Metric: The F1-score for this model is reported as 83.765%.

3. **ROC Curve:**

Visualization: A Receiver Operating Characteristic (ROC) Curve.

What it shows: This plot compares the performance of both models across all possible classification thresholds.

X-axis: False Positive Rate (FPR), which is the proportion of negative cases that are incorrectly classified as positive.

Y-axis: True Positive Rate (TPR), which is the proportion of positive cases that are correctly identified. This is also known as recall or sensitivity.

The dashed line: The diagonal line represents a random classifier (or a poor model), which has an Area Under the Curve (AUC) of 0.5.

**Model Comparison:**

The blue line represents Logistic Regression, with an AUC of 89.76%.

The green line represents Random Forest, with an AUC of 91.48%.

What the AUC means: The Area Under the Curve (AUC) is a measure of the model's ability to distinguish between classes. A higher AUC indicates a better model.

**Summary of Findings:**

By looking at all three visualizations, we can draw the following conclusions about the two models:

F1-Score: The Random Forest model has a higher F1-score (83.765%) than the Logistic Regression model (79.864%), indicating it has a better balance of precision and recall.

AUC Score: The Random Forest model also has a higher AUC (91.48%) than the Logistic Regression model (89.76%). This means that overall, the Random Forest model is better at separating the positive and negative classes.

Confusion Matrix Details: The Random Forest model correctly identified more true positives (921 vs. 837) and had fewer false negatives (146 vs. 230) compared to the Logistic Regression model. This suggests it is better at identifying the positive class (class 1).

In conclusion, all the visualizations consistently point to the Random Forest model performing better than the Logistic Regression model for this specific classification task.

# Task 2: Customer Segmentation Using Unsupervised Learning
Objective:
Cluster customers based on spending habits and propose marketing strategies tailored to each
segment.

Dataset:
Mall Customers Dataset

Instructions:
* Conduct Exploratory Data Analysis (EDA)
* Apply K-Means Clustering to segment customers
* Use PCA or t-SNE to visualize the clusters
* Suggest relevant marketing strategies for each identified segment

Results:
Here's a breakdown of what each visualization means:

1. **Genre Distribution:**

This is a bar plot (specifically, a countplot from the seaborn library).

It shows the distribution of customers by gender (Male and Female).

The visualization indicates that there are more female customers than male customers.

2. **Age Distribution:**

This is a histogram with a kernel density estimate (KDE) curve.

It shows the distribution of customer ages.

The plot suggests that the majority of customers are in their late 20s and early 30s. The curve shows a peak around this age range, and another smaller peak around 40s. The distribution is somewhat skewed to the left, indicating a younger customer base.

3. **Annual Income Distribution:**

Similar to the age distribution, this is a histogram with a KDE curve.

It shows the distribution of customer annual incomes (in thousands of dollars, or k$).

The plot shows that the highest frequency of customers have an annual income between $40k and $80k. The distribution appears roughly normal, centered around this range.

4. **Spending Score Distribution:**

This is another histogram with a KDE curve.

It shows the distribution of "Spending Score," a value between 1 and 100 assigned by the mall to customers based on their behavior.

The plot reveals that the spending scores are fairly well-distributed, with a peak around the 40-60 range. This suggests a large number of customers have a moderate spending score, with fewer at the very low or very high ends.

5. **Pairwise Relationships:**

This is a pair plot, which is a grid of plots showing the relationships between multiple variables.

The diagonal plots are kernel density plots for each variable (Age, Annual Income, and Spending Score), showing their individual distributions (similar to the single histograms shown earlier).

The off-diagonal plots are scatter plots showing the relationship between two variables:

Age vs. Annual Income: The scatter plot doesn't show a strong linear correlation, but there's a wider spread of income among younger adults (20s-30s) and a narrower, lower income range for older customers.

Age vs. Spending Score: This scatter plot reveals a few interesting patterns. Customers in their 20s and early 30s tend to have a wider range of spending scores, including some very high scores. The spending score seems to decrease with age.

Annual Income vs. Spending Score: This is a crucial plot for customer segmentation. It clearly shows distinct clusters of customers, which is a strong indicator that a clustering algorithm like K-Means would be effective. We can see clusters for high-income/low-spending, high-income/high-spending, low-income/low-spending, low-income/high-spending, and a large middle-of-the-road cluster. These clusters represent different customer segments.

6. **Gender vs. Key Features:**

This is a set of box plots.

They compare the distribution of Age, Annual Income, and Spending Score for Male and Female customers.

Age: The box plots show that the median age for both genders is similar, but the interquartile range (IQR) for females is slightly lower, suggesting they might, on average, be slightly younger than the male customers.

Annual Income: The median annual income for both genders is nearly identical, indicating no significant difference in income distribution between males and females in this dataset.

Spending Score: This is a key insight. The box plot shows that the median spending score for female customers is higher than for male customers. This suggests that, as a group, female customers tend to have higher spending scores.

These visualizations show the results of a customer segmentation analysis using a clustering algorithm, likely K-Means, on the mall customer data. They illustrate how customers were grouped into distinct segments and how those segments can be interpreted.

1. **Elbow Method for Optimal K:**

This plot is used to determine the optimal number of clusters for the K-Means algorithm.

The x-axis represents the number of clusters (k).

The y-axis represents the inertia, which is the sum of squared distances of samples to their closest cluster center.

The goal is to find the "elbow" point, where the rate of decrease in inertia slows down significantly. In this plot, the elbow appears to be at k = 5, suggesting that grouping the customers into 5 clusters is the most effective choice.

2. **Customer Segments (PCA & t-SNE):**

These scatter plots visualize the customer segments after applying dimensionality reduction techniques: Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).

Both plots show the customers grouped into 5 clusters, each represented by a different color. The goal of these visualizations is to confirm that the clusters are well-separated and distinct in a 2D space.

The PCA plot shows the data projected onto the first two principal components (PCA1 and PCA2), which capture the most variance.

The t-SNE plot is better for visualizing complex, non-linear relationships and shows a cleaner separation of the clusters. This confirms the clustering algorithm successfully identified distinct groups.

3. **Cluster Characteristics and Interpretation:**

The final set of visualizations provides a summary table and a textual description of each customer segment. This is the most crucial part of the analysis, as it gives meaning to the clusters.

Cluster 0 (Orange): Characterized by high average age (~55) and low average spending score (~42). These are likely "Broad value segment" shoppers, who respond to seasonal sales and coupons.

Cluster 1 (Red): This group has a high average annual income (~$86k) and a high average spending score (~81). These are the "High income & high spenders," who could be targeted with luxury bundles, concierge services, and VIP events.

Cluster 2 (Blue): This segment has a low average age (~26) and a high average spending score (~74). These are the "Young, active shoppers" or "High engagement shoppers" that respond to value packs, cashback, and installments.

Cluster 3 (Green): This group has a low average age (~27) and a low average spending score (~41). They are similar to Cluster 0 but younger, and also fall into the "Broad value segment."

Cluster 4 (Purple): Characterized by high average income (~$90k) but a very low average spending score (~18). These are the "High income but low spenders" who need to be targeted with personalized offers and loyalty perks to encourage them to spend more.

In summary, these visualizations complete the customer segmentation workflow by first identifying the optimal number of clusters, then visualizing the resulting groups, and finally providing a clear, actionable profile for each customer segment.

# Task 5: Interactive Business Dashboard in Streamlit
Objective:
Develop an interactive dashboard for analyzing sales, profit, and segment-wise performance.

Dataset:
Global Superstore Dataset

Instructions:
* Clean and prepare the dataset
* Build a Streamlit dashboard with filters (Region, Category, Sub-Category)
* Display key performance indicators (KPIs) using charts: Total Sales, Profit, Top 5 customers by sales

Results:
After running Streamlit app, here's a breakdown of what each visualization means:

**Key Performance Indicators (KPIs):**

* Total Sales: The total sales for the selected filters is $2,297,200.86.
* Total Profit: The total profit for the selected filters is $286,397.02.

**Sales & Profit by Category:**

This bar chart compares the total sales and profit for different product categories.

The categories displayed are Furniture, Office Supplies, and Technology.

For each category, there are two bars: one representing Sales (blue) and one representing Profit (orange).

From the chart, it appears that Technology has the highest sales and profit, followed by Office Supplies and then Furniture.

**Top 5 Customers by Sales:**

This horizontal bar chart identifies the top five customers based on their total sales.

The customers are listed on the y-axis, and their total sales are on the x-axis.

The top customer is Sean Miller, who has the highest sales.

The other customers in the top five are Tamara Chand, Raymond Buch, Tim Ashbrook, and Adrian Barton.

**Filters:**

The dashboard allows users to filter the data based on several criteria, which are visible on the left side of the screen. The currently selected filters are:

Region: East, West, South, and Central.

Category: Technology, Office Supplies, and Furniture.

Sub-Category: Accessories, Binders, Tables, Supplies, Phones, Chairs, Machines, Copiers, Appliance, Storage, Bookcases, Furnishings, Paper, Art, Envelopes, Labels and Fasteners.
   * Profit
   * Top 5 Customers by Sales

# Bank Marketing Project

This repo contains the code of the Bank Marketing Project on behalf of the Data Preparation, Exploration, and Visualization Course in ITI Intake 46.

## Project Overview

The goal of this project is to analyze the Bank Marketing dataset to understand the factors that influence whether a client subscribes to a term deposit. The analysis is divided into three levels: **Univariate**, **Bivariate**, and **Multivariate**.

---

## Business Questions

### Univariate Analysis

| #   | Question                                                        | Variable      | Chart Type         |
| --- | --------------------------------------------------------------- | ------------- | ------------------ |
| Q1  | What is the overall success rate of the campaign?               | `y`           | Bar / Pie          |
| Q2  | What is the age distribution of contacted clients?              | `age`         | Histogram + Boxplot|
| Q3  | What are the most common job types?                             | `job`         | Horizontal Bar     |
| Q4  | What is the distribution of education levels?                   | `education`   | Pie / Bar          |
| Q5  | How is call duration distributed?                               | `duration`    | Histogram          |
| Q6  | How many contacts are typically made per client?                | `campaign`    | Histogram          |
| Q7  | Which months have the highest call volume?                      | `month`       | Bar                |
| Q8  | What is the distribution of contact method (cellular vs telephone)? | `contact` | Pie                |
| Q9  | What is the distribution of marital status?                     | `marital`     | Bar                |
| Q10 | How many clients have housing loans?                            | `housing`     | Pie                |
| Q11 | How many clients have personal loans?                           | `loan`        | Pie                |
| Q12 | How many clients have credit in default?                        | `default`     | Pie                |
| Q13 | How is the balance distributed?                                 | `balance`     | Histogram          |
| Q14 | Which days have the highest call volume?                        | `day_of_week` | Histogram          |
| Q15 | How many clients subscribed in a term deposit before?           | `poutcome`    | Bar                |

---

### Bivariate Analysis

| #   | Question                                                              | Variables            | Chart Type                |
| --- | --------------------------------------------------------------------- | -------------------- | ------------------------- |
| Q16 | How does job type affect subscription rate?                           | `job` vs `y`         | Grouped Bar               |
| Q17 | Does marital status influence deposit subscription?                   | `marital` vs `y`     | Grouped Bar               |
| Q18 | Is education level correlated with subscription success?              | `education` vs `y`   | Grouped Bar               |
| Q19 | Does call duration relate to subscription outcome?                    | `duration` vs `y`    | Boxplot                   |
| Q20 | Do clients with housing loans subscribe more?                         | `housing` vs `y`     | Grouped Bar               |
| Q21 | Does contact method affect success rate?                              | `contact` vs `y`     | Bar                       |
| Q22 | Which month has the highest subscription rate?           | `month` vs `y`    | Line + Bar 
| Q23 | Does previous campaign outcome affect current subscription?           | `poutcome` vs `y`    | Bar                       |
| Q24 | How does age differ between subscribers and non-subscribers?          | `age` vs `y`         | Boxplot / Violin          |
| Q25 | Do clients with personal loans subscribe more?                        | `loan` vs `y`        | Bar                       |
| Q26 | Is there a difference in balance distribution by subscription outcome?| `balance` vs `y`     | Class Histogram           |

---

### Multivariate Analysis

| #   | Question                                                                    | Variables                              | Chart Type                   |
| --- | --------------------------------------------------------------------------- | -------------------------------------- | ---------------------------- |
| Q27 | Is the relationship between balance and success different across job types?| `balance` + `job` + `y`               | Grouped Boxplot / Facet Grid |
| Q28 | Is the relationship between duration and success different across job types?| `duration` + `job` + `y`               | Grouped Boxplot / Facet Grid |
| Q29 | Is the relationship between balance and success different across marital status?| `balance` + `marital` + `y`               | Grouped Boxplot / Facet Grid |
| Q30 | Is the relationship between age and success different across credit default?| `age` + `default` + `y`               | Grouped Boxplot / Facet Grid |
| Q31 | What is the correlation between all numeric features?                       | All numeric + `y`                      | Correlation Heatmap          |


---

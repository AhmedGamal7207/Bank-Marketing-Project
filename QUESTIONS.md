# Bank Marketing EDA — Questions & Answers

**Team**: Abdallah Mohamed, Ahmed Gamal, Ahmed Saleem  
**Track**: AI — Intake 46, Alexandria  
**Dataset**: Bank-Full.csv (45,211 records, 17 features)

---

## Univariate Analysis

### Q1 — What is the overall success rate of the campaign?

The dataset is heavily imbalanced. The majority of contacted clients did **not** subscribe to a term deposit. Only a small fraction responded positively, indicating that most marketing calls were unsuccessful.

---

### Q2 — What is the age distribution of contacted clients?

The age distribution is roughly bell-shaped with a right skew. Most clients contacted are between their mid-20s and mid-50s, with the bulk concentrated around the 30–40 age range. A smaller tail extends toward older clients (60+).

---

### Q3 — What are the most common job types?

The most frequently represented job types are **blue-collar**, **management**, and **technician** workers. These three categories collectively account for a large share of the contacted population, reflecting the bank's broad targeting strategy.

---

### Q4 — What is the distribution of education levels?

The majority of clients hold **secondary** education, followed by **tertiary**, then **primary**. A small portion have unknown education status. This suggests the bank's customer base skews toward moderately to well-educated individuals.

---

### Q5 — How is call duration distributed?

Call duration is right-skewed. Most calls are relatively short, but a meaningful number of calls extend significantly longer. This is notable because duration is strongly linked to subscription outcome — longer calls tend to correlate with successful subscriptions.

---

### Q6 — How many contacts are typically made per client?

The distribution of contacts per client (campaign variable) is heavily right-skewed. Most clients are contacted **1–3 times** during the campaign, and very few are contacted more than 5–6 times.

---

### Q7 — Which months have the highest call volume?

**May** has by far the highest call volume, followed by **July**, **August**, and **June**. The bank concentrates most of its campaign activity during the warmer months, with considerably fewer calls made in early and late months of the year.

---

### Q8 — What is the distribution of contact method (cellular vs telephone)?

**Cellular** is the dominant contact method, used for the majority of outreach. Telephone accounts for a smaller portion, and a notable share has **unknown** contact type, suggesting data quality issues in the contact channel field.

---

### Q9 — What is the distribution of marital status?

**Married** clients form the largest group, followed by **single**, then **divorced**. The distribution reflects typical demographics of a working-age banking customer base.

---

### Q10 — How many clients have housing loans?

Roughly **half** of the clients have a housing loan, making it an evenly split feature. This variable turns out to have a meaningful relationship with subscription behavior.

---

### Q11 — How many clients have personal loans?

A minority of clients have personal loans — the **majority do not**. Personal loan status, similar to housing loans, negatively affects subscription likelihood.

---

### Q12 — How many clients have credit in default?

Very few clients are in credit default — the vast majority have no default history. This makes the variable highly imbalanced and less directly informative as a standalone predictor.

---

### Q13 — How is the balance distributed?

Balance is right-skewed with a long tail. Most clients hold modest account balances, but a small number of clients have very high balances. There are also some clients with negative balances (debt).

---

### Q14 — Which days have the highest call volume?

Call volume is relatively spread across the working days of the month, with no single day showing a dramatically dominant pattern. Mid-month days tend to see slightly higher activity.

---

### Q15 — How many clients subscribed to a term deposit before?

The majority of clients have an **unknown** previous outcome (indicating they had no prior campaign contact). Among those with a known outcome, a small group previously succeeded and a larger group previously failed to subscribe.

---

## Bivariate Analysis

### Q16 — How does job type affect subscription rate?

Job type significantly influences subscription likelihood. **Students** and **retired** clients show notably higher subscription rates compared to blue-collar or service workers. This suggests targeting specific occupational groups could improve campaign ROI.

---

### Q17 — Does marital status influence deposit subscription?

Yes. **Single** clients subscribe at the highest rate (**14.95%**), followed by **divorced** (**11.95%**), then **married** (**10.12%**). Single clients may have fewer financial obligations, making them more willing to commit to a term deposit.

---

### Q18 — Is education level correlated with subscription success?

Yes. Subscription rates increase with education level:
- Primary: **8.63%**
- Secondary: **10.56%**
- Tertiary: **15.01%**
- Unknown: **13.57%**

Clients with tertiary education are nearly twice as likely to subscribe compared to those with primary education.

---

### Q19 — Does call duration relate to subscription outcome?

Yes, strongly. Clients who subscribed have significantly **longer call durations** than non-subscribers. This is one of the most discriminating features in the dataset. However, duration is only known after the call ends, so it cannot be used as a pre-call predictor.

---

### Q20 — Do clients with housing loans subscribe more?

No. Clients **without** housing loans subscribe at a rate of **16.70%**, compared to only **7.70%** for those with housing loans. Having a housing loan appears to be a financial burden that reduces willingness to commit to a term deposit.

---

### Q21 — Does contact method affect success rate?

Yes. **Cellular** contacts yield a higher subscription rate than telephone contacts. Unknown contact type performs worst. This suggests that mobile outreach is more effective and worth prioritizing.

---

### Q22 — Which month has the highest subscription rate?

Despite **May** having the highest call volume, months like **March**, **September**, **October**, and **December** show the **highest subscription rates**. This reveals an inverse relationship — months with fewer calls tend to have better conversion, possibly because contacts are more targeted during those periods.

---

### Q23 — Does previous campaign outcome affect current subscription?

This is the **strongest bivariate predictor**. Clients who **previously subscribed** convert at a rate of **64.73%**, compared to:
- Previously failed: **12.61%**
- Unknown: **9.16%**

Prior success is a near-decisive indicator of future subscription.

---

### Q24 — How does age differ between subscribers and non-subscribers?

Subscribers tend to be slightly **older on average** and also show a concentration among **younger clients (under 30)**. The middle-aged group (30–50) has the lowest subscription rate. This U-shaped pattern suggests both young and retired clients are more receptive.

---

### Q25 — Do clients with personal loans subscribe more?

No. Clients **without** personal loans subscribe at **12.66%**, while those with personal loans subscribe at only **6.68%**. Like housing loans, personal debt reduces subscription likelihood.

---

### Q26 — Is there a difference in balance distribution by subscription outcome?

Yes. Clients who subscribed tend to have **higher account balances** on average. The balance distributions for subscribers and non-subscribers overlap but subscribers skew toward higher values, suggesting financial stability is a positive signal.

---

## Multivariate Analysis

### Q27 — Is the relationship between balance and success different across job types?

Yes. The balance-subscription relationship varies by occupation. **Retired** and **management** clients who subscribe tend to have higher balances compared to subscribers in other job categories. For blue-collar workers, the balance gap between subscribers and non-subscribers is smaller.

---

### Q28 — Is the relationship between duration and success different across job types?

Yes. Longer call durations consistently predict subscription across all job types, but the magnitude differs. **Students** and **management** clients who subscribe show particularly long durations, suggesting these groups require more engagement before committing.

---

### Q29 — Is the relationship between balance and success different across marital status?

Yes. For all marital status groups, subscribers tend to have higher balances. **Single** subscribers show the widest balance advantage, while **divorced** subscribers show the narrowest gap between subscribers and non-subscribers.

---

### Q30 — Is the relationship between age and success different across credit default status?

Yes. Among clients **without credit default**, both younger (under 30) and older (over 60) clients are more likely to subscribe. Among the rare clients **with credit default**, the age-subscription relationship is weaker and noisier due to the small sample size.

---

### Q31 — What is the correlation between all numeric features?

The correlation heatmap shows:
- **Duration** has the highest positive correlation with subscription (`y`)
- **Pdays** and **previous** have mild positive correlations
- **Campaign** (number of contacts) has a slight negative correlation
- **Age**, **balance**, and **day** show weak correlations with the target
- Most numeric features show low inter-feature correlation, meaning they carry relatively independent information

---

## Conclusion

This exploratory analysis of the Bank Marketing dataset reveals several key patterns that drive term deposit subscription:

**1. Prior campaign success is the strongest predictor.**  
Clients who subscribed in a previous campaign convert at 64.73% — far above all other groups. Prioritizing re-engagement of past subscribers should be the top campaign strategy.

**2. Financial obligations reduce subscription likelihood.**  
Clients with housing loans or personal loans subscribe at roughly half the rate of those without. These clients face competing financial pressures that make long-term commitments less attractive.

**3. Education and occupation shape receptiveness.**  
Higher-educated clients and those in student or retired segments respond better to the campaign. Tailoring messaging to these segments could significantly improve conversion rates.

**4. Cellular outreach and longer calls signal success.**  
Cellular contacts outperform telephone, and successful calls tend to run longer. While call duration cannot be used predictively, it confirms that deeper conversations lead to better outcomes — agents should be trained to sustain engagement.

**5. Campaign timing matters.**  
Although May generates the most calls, months like March, September, October, and December yield the highest subscription rates. The bank may benefit from redistributing campaign effort toward these higher-converting months.

**6. Demographic sweet spots exist at the extremes of age.**  
Both young adults and retirees are more likely to subscribe than middle-aged clients. Marketing strategies should be differentiated by life stage, as motivations and financial priorities differ substantially across age groups.
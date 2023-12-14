# Convex Separators
A Convex Separator is defined as an intersection of half-spaces that separates one class from another. We have built a method to learn Convex Separators in a stable way. <br />
Contributors: <br />
Amit Prasad (https://www.linkedin.com/in/amit-prasad-951081119/) <br />
Prof. Rahul Garg(https://www.cse.iitd.ac.in/~rahulgarg/) <br />
Prof. Yogish Sabharwal (https://www.cse.iitd.ac.in/~yogish/HomePage/Home.html) <br />

Following are some synthetic datasets for which the separator is shown in action:
![image](https://github.com/Amit-Prasad/Convex-Separators/assets/22973646/dbe04a00-22d9-4c3c-841a-ead28ae36859)

It can also do various complicated patterns like 
![image](https://github.com/Amit-Prasad/Convex-Separators/assets/22973646/99b1c48b-0e0f-425d-9d7a-cecf96a68941)
![image](https://github.com/Amit-Prasad/Convex-Separators/assets/22973646/2d49bd3f-a2c2-4919-bd1f-abbd9019f663)

Performance on real datasets are in terms of AUROC scores here
|Algorithms            | Shoppers | Spambase | Telco Churn | Churn | Diabetes |
|----------------------|----------|----------|-------------|-------|----------|
|Convex Separators     |     0.92 |     0.97 |    0.85     |  0.79 | 0.85     |
|Logistic Regression   |     0.89 |     0.97 |    0.85     |  0.75 | 0.79     |
|MLP2                  |     0.91 |     0.98 |    0.62     |  0.86 | 0.85     |
|MLP3                  |     0.91 |     0.98 |    0.82     |  0.85 | 0.85     |
|Random forest         |     0.92 |     0.99 |    0.81     |  0.82 | 0.85     |
|MLP1                  |     0.92 |     0.98 |    0.81     |  0.86 | 0.86     |
|XGBoost               |     0.92 |     0.99 |    0.8      |  0.83 | 0.8      |



# **Federated Learning in the Clinical Environment**

Federated Learning (FL) is a cutting-edge technique that preserves data privacy while leveraging sensitive information, such as patient medical data, to train and improve Machine Learning (ML) models.

This project implements a proof-of-concept FL framework designed for use in a clinical environment, with the goal of building ML models to enhance patient care. Specifically, the project focuses on using MRI images of brains with four different classifications of tumors. TensorFlow was used to create the ML models, while the Flower framework was employed for the FL methods.

Our results demonstrated that the FL model accuracy was within 10% of the centralized model's accuracy. Furthermore, under stricter data privacy conditions, there was noticeable improvement, indicating that FL is a promising method for clinical applications.

---

## **Installation**

This project is implemented in Python 3. To install the required Python packages, run:

```bash
pip install -r requirements.txt


## Running the code (Federated Learning)

Note: A Linux environment is necessary to run this code (either a VM or WSL will work)

1) In a terminal, start the central server using:
```bash
python3 server.py
```

2) Next, open up 4 more terminals, one for each of the 4 Clients. Start each client in its own terminal using the commands:
```bash
python3 client.py 0
python3 client.py 1
python3 client.py 2
python3 client.py 3
```

The Federated Learning process will only begin once all 4 clients are connected.

After a brief pause, the server should start sampling the clients, who will each be training Machine Learning models!

## Running the code (Machine Learning)

All the code for our individual Machine Learning Models is located in the Jupyter Notebook: mri_classification.ipynb


## Authors
- Coral, G
- Hait, J
- Isaak, K
- Watkins, A#   m a j o r - p r o j e c t 
 
 

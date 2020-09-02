# COVID-GAN
As a result of the COVID-19 epidemic in the United States, there has been a tremendous increase in the number of patients seeking primary COVID-19 screenings. 
This influx of patients and increase in demand for testing has overloaded hospitals, thereby creating excessive burden for healthcare workers. One of the primary means of diagnosing 
COVID-19 is through visual examination of chest X-ray scans. 

![Image of Chest X-Ray](https://github.com/billsun9/COVID-GAN/blob/master/cnn/augmented_imgs/xray_0_1692.jpeg)
In this project, a relatively shallow convolutional neural network (CNN) was developed to diagnose COVID-19 from chest X-ray scans. The CNN was trained on the [Kaggle COVID-19 Radiography
Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database), which contains 219 COVID-19 positive images, 1341 normal images and 1345 viral pneuomonia images. After
training for 50 epochs, the final model achieved ~97.6% accuracy.

However, there a distinct scarcity of openly available radiological images for COVID-19, so a [Generative Adversarial Network (GAN)](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
was developed to augment existing image datasets.

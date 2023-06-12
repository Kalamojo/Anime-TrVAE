# Anime Transfer Variational Autoencoder

Image Transferring by way of a Variational Autoencoder

## Model Demo

Here is the link to the tensorflow model: https://drive.google.com/file/d/1RXZmK8eS1m2d3ZEje0e2me5uYccirFuf/view?usp=sharing

To run an image transferring demo, download the model and add it to your local repository. Then, simply install the requirements:

> Please note that the demo only supports python version 3.9 (and maybe earlier)

```bash
pip install -r requirements.txt
```

Then, run the streamlit demo:

```bash
streamlit run demo.py
```

## Introduction

Filters for images already exist, and have been around for quite a while. Applications like Snapchat make use of them, and are able to superimpose items on users, change the coloring/light of an image, or even transfer the style of a person's face. The way these applications generally work are by the use of Generative Adversarial Networks (GANs). The GAN approach generally involves training a model to slightly modify an input image until it resembles images from an anime. While this method effectively accomplishes the bare minimum task, it really doesn't truly generate an anime version of a realistic image.

Our proposed method makes use of a Transfer Variational Autoencoder (trVAE) proposed in a transfer learning research paper (Lotfollahi et al). This model transfers images in a generative fashion, and can easily be supplied labels for transferring instructions. Furthermore, we hope to display how novel methods can be applied to unique problems in general.

To allow others to actually make use of our model, we plan to create a website application for image transferring. It will provide image and label selection functionality, and should return results within a couple of seconds. On this site, we also plan to document the model architecture along with our data, model parameters, and of course, the original paper and repository from which we obtained the model.

## Materials and Methods

Autoencoders simply take data, encode it, and then decode it. While this seems like a useless task at first, the main objective of using an Autoencoder is to train the encoder to effectively reduce data dimensionality. This reduced data ideally contains all of the information from the original data, yet is more effective for training
and susceptible to change.

The Transfer Variational Autoencoder takes advantage of this feature of encoded data with the following method:

1. Data is encoded
2. Encoded data is manipulated
3. Data is decoded

![alt text](https://github.com/Kalamojo/Anime-TrVAE/blob/main/images/diagram.png?raw=true)
:---------------------------:
trVAE architecture diagram

As the diagram depicts, the exact method of manipulating encoded data is with a labelling system. Encoded data is concatenated with an initial label, and encoded data is combined with a second label. The encoder portion of the model effectively learns that data will always be “supplied” a label later on, so with this system, it learns to strip its data of its label class. And once data has been stripped of its class, it is primed and ready to be supplied any second label, effectively transferring data from one class to any other.

![alt text](https://github.com/Kalamojo/Anime-TrVAE/blob/main/images/transfer_learning_visialization.png?raw=true)

## Results

Original         |  Transferred
:---------------------------:|:---------------------------:
![alt text](https://github.com/Kalamojo/Anime-TrVAE/blob/main/images/female_real_org.jpg?raw=true) | ![alt text](https://github.com/Kalamojo/Anime-TrVAE/blob/main/images/female_anime_transfer.jpg?raw=true)
![alt text](https://github.com/Kalamojo/Anime-TrVAE/blob/main/images/male_real_org.png?raw=true) | ![alt text](https://github.com/Kalamojo/Anime-TrVAE/blob/main/images/male_anime_transfer.jpg?raw=true)
![alt text](https://github.com/Kalamojo/Anime-TrVAE/blob/main/images/female_anime_org.jpg?raw=true) | ![alt text](https://github.com/Kalamojo/Anime-TrVAE/blob/main/images/female_real_transfer.jpg?raw=true)
![alt text](https://github.com/Kalamojo/Anime-TrVAE/blob/main/images/male_anime_org.jpg?raw=true) | ![alt text](https://github.com/Kalamojo/Anime-TrVAE/blob/main/images/male_real_transfer.jpg?raw=true)

## Areas of Improvement

- Enable whole-body transfers, not just faces
- Diversify training datasets for more robust transfers
- Host model somewhhere for website deployment
- Add more classes of image transferring

## Acknowledgments

- Lotfollahi, M., Naghipourfar, M., Theis, F. J., & Wolf, F. A. (2020). Conditional out-of-distribution generation for unpaired data using transfer VAE. Bioinformatics (Oxford, England), 36(Suppl_2), i610–i617. https://doi.org/10.1093/bioinformatics/btaa800
- Liu et al (2015). Deep Learning Face Attributes in the Wild https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Hugging Face Anime Face Dataset https://huggingface.co/datasets/DrishtiSharma/Anime-Face-Dataset
- Google Cartoonset https://google.github.io/cartoonset/
- UTRGV Deep Learning professor Dr. Kim’s GPU Clusters

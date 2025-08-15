# ChatBot
'''
I built a Stage-1 text-to-image generation model based on a conditional GAN architecture. The model takes text embeddings from bird image captions (CUB_200_2011 dataset) and a random noise vector, then generates 64×64 color images matching the description.
The workflow includes:
Data loading & preprocessing — crops birds using bounding boxes, resizes to 64×64, normalizes to the GAN’s input scale.
Embedding compressor — reduces 1024-dimensional text embeddings to 128-dimensional conditioning vectors.
Generator network — progressively upsamples from 4×4 to 64×64 resolution, conditioned on the text embedding + noise.
Discriminator network — judges real vs. generated images, also conditioned on text embeddings.
Custom KL loss — regularizes the latent conditioning vector for better generation diversity.
Training loop — alternates between updating the discriminator on real/fake/wrong-caption pairs and updating the generator via the adversarial + KL loss.
Image saving — exports generated samples every few epochs to track training progress.
Tech Stack:
Python, TensorFlow/Keras
Numpy, Pandas, Pillow, Matplotlib
CUB_200_2011 dataset + precomputed char-CNN-RNN text embeddings

The trained Stage-1 generator can produce small but coherent bird images conditioned on textual descriptions, forming the base for a potential Stage-2 model to refine them to higher resolutions.
'''

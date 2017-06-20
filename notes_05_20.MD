**Nishant**

I found out
why your VAE was not working
It was related to data preprocessing
And a fault in imshow
I normalised it to [0,1] and used grey map
Maybe I have got some results
I will send you a screenshot

input image:
![input_image](https://lh3.googleusercontent.com/-aSJEYSWmhxg/WUhdWc1aAAI/AAAAAAAAAVM/uzZYubRUCpkWEG2Pc5khyGZpIGriaq41wCK8BGAs/s512/2017-06-19.png)
and the output is:
![output_mse](https://lh3.googleusercontent.com/-4gDqKwshn-A/WUhdcZOmXnI/AAAAAAAAAVQ/wUovnzYLVEIjafHr1jYMFQ5TdlY5RiwvgCK8BGAs/s512/2017-06-19.png)
I trained for 30 epoch and as this was mean squared error so it seems blurry

For binary cross entropy:
![output_bce](https://lh3.googleusercontent.com/-9dUVSWVJggg/WUheTEiB5PI/AAAAAAAAAVY/1AzvnqXPrboR_ff_5AnuGiX0iGgF34iRQCK8BGAs/s512/2017-06-19.png)
Params:

batch_size = 10
latent_dim = 10
intermediate_dim = 60
epsilon_std = 1.0


**Tue, 1:30 AM
Gagan Narula (gagoburger@gmail.com)**

hmm.. so good work, but there are a few things to think about:
1. When the VAE is trained, we want to only sample from it, so basically choose some noisy Z values and generate the data.. i want to see to what kinds of features come out.
2. I don't think binary cross entropy makes a lot of sense.. u see the binomial generative model (last layer of decoder) is used to model data that bernoulli generated .. like in mnist, where each pixel has value 0 or 1 not between them. The picture is nicer of course, less blurry as you show, but i don't know to what extent it is c
it is correct
3. In params, maybe you will need to increase the latent space dimensionality to 16-20 and what do you mean by intermediate_dim ? how many layers do u have? 
4. Lets try to replace the encoder and decoder with two things: 1. All convolutional layers with leaky relu's 2. Replace encoder and decoder with LSTM, then use minibatches of data which are temporally ordered (like a sequence of spectrogram columns instead of single randomly sorted columns as of now) .. we could use the last value of the last out of the encoder to estimate the mu and sigma of Z values and the last state of the decoder from an initial value of Z to generate the data through encoder
But LSTM version can come later.
First thing you should do is make a document in the git folder and copy paste this conversation including your first results into it.

**Nishant**
Yeah sure!
And I will sample from latent dimension and show you the output
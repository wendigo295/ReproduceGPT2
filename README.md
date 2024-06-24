I basically built and trained the GPT-2 model from scratch in Pytorch watching Andrez Karpathy's Video "Let's Reproduce GPT-2". I understood and wrote the code from scratch and trained the model on the FineWeb(EDU) dataset on a batch size of 0.5M tokens and evaluated it by the HellaSwag dataset which is also used in the GPT3 paper on 4 epoch. The trained model surpassed the original GPT-2(124M) model and almost reach the accuracy of GPT-3 in evaluation.

Note that GPT-2 and GPT-3 are both simple language models, trained on internet documents, and all they do is "dream" internet documents. So this repo/video this does not cover Chat finetuning, and you can't talk to it like you can talk to ChatGPT. The finetuning process (while quite simple conceptually - SFT is just about swapping out the dataset and continuing the training) comes after this part and will be covered at a later time. For now this is the kind of stuff that the 124M model says if you prompt it with "Hello, I'm a language model," after 10B tokens of training:

```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
Hello, I'm a language model, and I'm an English person. In languages, "speak" is really speaking. Because for most people, there's
```

And after 40B tokens of training:

```
Hello, I'm a language model, a model of computer science, and it's a way (in mathematics) to program computer programs to do things like write
Hello, I'm a language model, not a human. This means that I believe in my language model, as I have no experience with it yet.
Hello, I'm a language model, but I'm talking about data. You've got to create an array of data: you've got to create that.
Hello, I'm a language model, and all of this is about modeling and learning Python. I'm very good in syntax, however I struggle with Python due
```

## License

MIT

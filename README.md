I built and trained the GPT-2 model from scratch in Pytorch watching **Andrez Karpathy's** Video "**Let's Reproduce GPT-2**". I understood and wrote the code from scratch trained the model on the **FineWeb(EDU)** dataset on a batch size of **0.5M** tokens and evaluated it by the **HellaSwag** dataset which is also used in the GPT3 paper on 4 epochs. The trained model **surpassed** the original **GPT-2(124M)** model and almost reached the accuracy of **GPT-3** in evaluation.

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
𝐊𝐞𝐲 𝐏𝐨𝐢𝐧𝐭𝐬:

𝐍𝐞𝐭𝐰𝐨𝐫𝐤 𝐒𝐤𝐞𝐥𝐞𝐭𝐨𝐧 𝐂𝐨𝐧𝐬𝐭𝐫𝐮𝐜𝐭𝐢𝐨𝐧: Constructed the entire network skeleton to match the schema of GPT-2 as used in Hugging Face transformers.

𝐒𝐡𝐚𝐫𝐞𝐝 𝐖𝐞𝐢𝐠𝐡𝐭 𝐌𝐚𝐭𝐫𝐢𝐱: Shared the same weight matrix between the embedding layers and the pre-softmax linear transformation, as specified in the GPT-2 paper.

𝐎𝐩𝐭𝐢𝐦𝐢𝐳𝐞𝐝 𝐓𝐫𝐚𝐢𝐧𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐓𝐅𝟑𝟐: Training is optimized by setting matrix multiplication to the TF32 datatype, which offers the range of FP32 with the precision of FP16, resulting in an 8x speedup.

𝐀𝐮𝐭𝐨𝐦𝐚𝐭𝐢𝐜 𝐌𝐢𝐱𝐞𝐝 𝐏𝐫𝐞𝐜𝐢𝐬𝐢𝐨𝐧: Further optimization is achieved using automatic mixed precision with 𝐭𝐨𝐫𝐜𝐡.𝐚𝐮𝐭𝐨𝐜𝐚𝐬𝐭, improving efficiency to some extent.

𝐓𝐫𝐚𝐢𝐧𝐢𝐧𝐠 𝐎𝐩𝐭𝐢𝐦𝐢𝐳𝐚𝐭𝐢𝐨𝐧 𝐰𝐢𝐭𝐡 𝐭𝐨𝐫𝐜𝐡.𝐜𝐨𝐦𝐩𝐢𝐥𝐞: Training is additionally optimized by using 𝐭𝐨𝐫𝐜𝐡.𝐜𝐨𝐦𝐩𝐢𝐥𝐞, providing a 2x speedup. Karpathy explains this process via the Memory Hierarchy, focusing on Bandwidth and Memory Size.

𝐅𝐥𝐚𝐬𝐡 𝐀𝐭𝐭𝐞𝐧𝐭𝐢𝐨𝐧 𝐓𝐞𝐜𝐡𝐧𝐢𝐪𝐮𝐞: Employed the Flash Attention technique, which avoids materializing large attention matrices and reads and writes to slow GPU HBM, resulting in a 7.6x speedup in attention computation.

𝐆𝐫𝐚𝐝𝐢𝐞𝐧𝐭 𝐀𝐜𝐜𝐮𝐦𝐮𝐥𝐚𝐭𝐢𝐨𝐧 𝐓𝐞𝐜𝐡𝐧𝐢𝐪𝐮𝐞: Implemented gradient accumulation technique, allowing for training with huge batch sizes (up to 0.5M tokens) without overwhelming the GPU.

𝐃𝐢𝐬𝐭𝐫𝐢𝐛𝐮𝐭𝐞𝐝 𝐃𝐚𝐭𝐚-𝐏𝐚𝐫𝐚𝐥𝐥𝐞𝐥 𝐓𝐫𝐚𝐢𝐧𝐢𝐧𝐠: Finally used Distributed Data-Parallel, making the necessary code adjustments and providing thorough explanations.

𝐅𝐢𝐧𝐞𝐖𝐞𝐛 𝐃𝐚𝐭𝐚𝐬𝐞𝐭: Used the FineWeb(EDU) dataset, a filtered version of Common Crawl tailored to high-quality educational subsets for training.

𝐄𝐯𝐚𝐥𝐮𝐚𝐭𝐢𝐨𝐧 𝐰𝐢𝐭𝐡 𝐇𝐞𝐥𝐥𝐚𝐒𝐰𝐚𝐠: For evaluation, Used HellaSwag, as utilized in the GPT-3 paper, surpassing GPT-2 and achieving almost GPT-3 level accuracy in just 4 epochs.

## License

MIT

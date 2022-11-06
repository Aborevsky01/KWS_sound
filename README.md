# ASR_sound

## Brief report  

* #### Description and result of each experiment

While more broad descriptions can be found in the notebook itself, here we present only the key facts.

1. Experiment **"Low and Everything"**
    
    **A**. QAT + Dark KD + Attention KD + qint8
    
    **B**. Speed Rate 9.38
    
    **C**. Compression rate 8.26
    
    **D**. Quality 
  
2. Experiment **"Stupid student, Smart teacjer"** (Winner)
    
    **A**. QAT + Dark KD + Attention KD + RNN KD + qint8
    
    **B**. Speed Rate 10.1
    
    **C**. Compression rate 8.81
    
    **D**. Quality 4.6e-5

3. Experiment **"What if..?"**
    
    **A**. Dark KD + Attention KD + qint8
    
    **B**. Speed Rate 3.5
    
    **C**. Compression rate 2.6
    
    **D**. Quality 5e-5
    
4. Experiment **"MC dark KD"**
    
    **A**. Dark KD 
    
    **B**. Speed Rate 5.7
    
    **C**. Compression rate 5
    
    **D**. Quality 5.4e-5


* #### How to reproduce your model?

All the required code and hyperparameters' values can be found in cells, devoted to each experiment. However, it is essential to underscore that models hardly rely on the chosen seed's value. Nevertheless, it sometimes might take quite more / less epochs than stated, so special condition to exit training loop at sufficient quality was added.

However, mainly we take T = 10 and high value of alpha for dark KD, train for approximate 15-20 epochs with lr starting from 3e-3, going down to 2e-4 in step manner. 

* Training logs for each experiment are all placed in the respective directory.

* What worked and what didn't work?

>  pros

One of the key finding was dark KD, since it was really enhancing the student model in first experiment, providing quality bellow the benchmark. However, there were issues with it in the second experiment with RNN: it was hard to identify the needed alpha. Nevertheless, it is included in each experiment, being the core of training. 

 Another point to mention is Quantization Aware training (QAT). The base model + QAT suddenly showed really fascinating result, which in turn made training with it as a teacher more productive. At the same time, while it wasn't really killing the quality of student, having limited fascilities concerning layers available for quantization, QAT was applied only to Conv - ReLU. While there were plans to introduce it also to linear layers, they have never become true.

 Great tool for compression appeared to be dynamic quantization. And among available options, qint8 has performed outstanding in terms of reducing model's size. However, quite rational consequence was fall of quality above the benchmark, so it should have been used really carefully.



> cons

On the other hand, if we continue topic of dynamic quantization, float16 seemed to be quite controversial. From one point, it really contained the announced before quantization quality. At the same time, it did not reduce size on enough level, if we think of really high rates. Nevertheless, it seems to be a save way to quantize without being afraid for a severe quality fall.

Also to add is pruning. This tool really tried it's best to bleed me dry. First of all, it appered that we need to wrap torch structured pruning with in order to make model drop the annihilated layers. In other words, since reducing values to zero does not change either FLOPs or size, we had to create new model after each block's pruning. However, it wouldn't be so bad if we didn't face the following challenge. The quality was dropping in crazy manner compared to processing without prunning. Thus, it is not presented even in a single experiment, but we have a corresponding example.

Last to mention is the usage of RNN insted of GRU. While some collegues asserted there is a way to train it for the benchmark, we did not succeed in it, at least for good rates of speed and compression for a very long time. Only a really hard search for hyperparameters has accidently given us the desired quality.

* What were the major challenges?

1. torch library implementation of QAT appeared to be ambiguous and partially a real rocket science. While they have an outstanding lecture (I also recommend to share it for future years), still it's embodiement was not as easy as it could be imagined. Nevertheless, it worked and thanks God for it. Additionally, it could not be used with CUDA, which is makes calculations more complicated.

2. Pruning was already widely discussed, but not to mention the very poor quality of implementation suggested is a true crime.

3. Streaming was quite an obstacle on this path. StreamReader was not an easy one to install, however, as it started working, the effect was truly amazing! Recording drom microphone in this homework seems to be vital for topic's understanding.

4. Well, of course it is part of Deep Learning, but still such a strong randomness of overcoming benchmark, correlating only with seed, frustrates during the first days of working.

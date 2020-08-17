# Multimodal_Classification_Co_Attention
Multimodal classification solution for the SIGIR(Special interest group for information retrieval) eCOM 2020 - https://sigir-ecom.github.io/index.html    using Co-attention and transformer language models. The paper with the approach and technique was presented a is available at -  https://sigir-ecom.github.io/ecom20DCPapers/SIGIR_eCom20_DC_paper_7.pdf

The architecture of the proposed model can be viewed as shown below:

![SIGIR_ARCHITECTURE](https://github.com/VarnithChordia/Multimodal_Classification_Co_Attention/blob/master/Architecture_3.png)




## 1. Preparations

Clone this repository:

```
git clone https://github.com/VarnithChordia/Multimodal_Classification_Co_Attention.git
```

## 2. Install the requirements.txt into the conda environment or another virtual enviroment

```
pip install -r requirements.txt
```

Better use: NVIDIA GPU + CUDA9.0 + Pytorch 1.0.1


## 3.
Train the model an example to do so 

```
Python3 main.py --mode=‘train’ --model = 'camembert-base' --model_type= ‘camem’
--embs=768 --self_attention = False --train_img=‘<insert image directory>’ --train_file=‘<training dataset file(csv) containing the product information>’ --labels_file =‘<insert the labels file>’
--num_cls=27 --batch_size=16 --num_epochs=10 --lr=.001 --output_model_name=‘<insert model name>’ --output_model_path=‘<insert model path>’ --requires_grad=True
```


This placed 2nd in the overall competition.

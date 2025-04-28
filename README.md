# Hierarchical Multi-level Prototypes (HierProtoPNet)

### Pytorch implementation for the IEEE JBHI paper "[Progressive Mining and Dynamic Distillation of Hierarchical Prototypes for Disease Classification and Localisation](https://ieeexplore.ieee.org/abstract/document/10955117)"
Email: chongwangsmu@gmail.com.

## Introducation:

medical image analysis tasks need to handling the complexity of diverse lesion characteristics.

<div align=center>
<img width="630" height="275" src="https://github.com/cwangrun/HierProtoPNet/blob/master/img/intro.png"/></dev>
</div>

## Method:
This appraoch leverages hierarchical visual prototypes across different semantic feature granularities to effectively capture diverse lesion patterns. 
To increase utility of the prototypes, we devise a prototype mining paradigm to progressively discover semantically distinct prototypes, offering multi-level complementary analysis of complex lesions. 
Also, we introduce a dynamic knowledge distillation strategy that allows transferring essential classification information across hierarchical levels, thereby improving generalisation performance. 

<div align=center>
<img width="900" height="240" src="https://github.com/cwangrun/HierProtoPNet/blob/master/img/arch.png"/></dev>
</div>


## Datasets:
1. Mammographic images ([CSAW-S](https://github.com/ChrisMats/CSAW-S))
2. OCT images ([NEH](https://data.mendeley.com/datasets/8kt969dhx6/1))
3. NIH chest X-rays ([NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data))


## Training/Testing:
1. Run python main.py to train the model and evaluate its disease diagnosis accuracy. Our trained models are provided at [ChestX-ray14](https://drive.google.com/file/d/1svxfab5YG2BVoSKe99krhwWeqQyQFUqw/view?usp=drive_link) and [ODIR](https://drive.google.com/file/d/1ykIhO6d2AqFO0Wy4Rmr4VIzvTVeoQIaQ/view?usp=drive_link):
2. Each prototype is visualized as the nearest non-repetitive training patch representing its corresponding disease class using push.py.


## Prototype visualisation:
CIPL leverages disentangled class prototypes, learned from the training set, as anchors for diagnostic reasoning.
To understand the decision process for a given test image, run interpretable_reasoning.py. 
This will generate a set of similarity (activation) maps that highlight the correspondence between the test image and the prototypes of each disease class, providing insights into the model's reasoning.

<div align=center>
<img width="900" height="325" src="https://github.com/cwangrun/HierProtoPNet/blob/master/img/prototypes.png"/></dev>
</div>



## Lesion/disease Localisation:
CIPL demonstrates high-quality visual prototypes that are both disentangled and accurate (aligning well with actual lesion signs), outperforming previous studies. For further details, please refer to our paper.

<div align=center>
<img width="880" height="470" src="https://github.com/cwangrun/HierProtoPNet/blob/master/img/mammo.png"/></dev>
</div>

<div align=center>
<img width="650" height="550" src="https://github.com/cwangrun/HierProtoPNet/blob/master/img/chestxray.png"/></dev>
</div>



## Citation:
```
@article{wang2025progressive,
  title={Progressive Mining and Dynamic Distillation of Hierarchical Prototypes for Disease Classification and Localisation},
  author={Wang, Chong and Liu, Fengbei and Chen, Yuanhong and Kwok, Chun Fung and Elliott, Michael and Pena-Solorzano, Carlos and McCarthy, Davis James and Frazer, Helen and Carneiro, Gustavo},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
```

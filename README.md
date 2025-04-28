# Hierarchical Multi-level Prototypes (HierProtoPNet)

### Pytorch implementation for the IEEE JBHI paper "[Progressive Mining and Dynamic Distillation of Hierarchical Prototypes for Disease Classification and Localisation](https://ieeexplore.ieee.org/abstract/document/10955117)"
Email: chongwangsmu@gmail.com.

## Introducation:

Medical image analysis tasks need to handle the complexity of diverse lesion characteristics: considerable size, shape, and appearance variations of the lesion structures from the same disease class.

<div align=center>
<img width="630" height="275" src="https://github.com/cwangrun/HierProtoPNet/blob/master/img/intro.png"/></dev>
</div>

## Method:
This appraoch leverages hierarchical visual prototypes across multiple semantic feature granularities to effectively capture diverse lesion patterns. 
To increase utility of the prototypes, we devise a prototype mining paradigm to progressively discover semantically distinct prototypes, offering multi-level complementary analysis of complex lesions. 
Also, we introduce a dynamic knowledge distillation strategy that allows transferring essential classification information across hierarchical levels, thereby improving generalisation performance. 

<div align=center>
<img width="900" height="240" src="https://github.com/cwangrun/HierProtoPNet/blob/master/img/arch.png"/></dev>
</div>


## Datasets:
1. Mammographic images ([CSAW-S](https://github.com/ChrisMats/CSAW-S))
2. OCT images ([NEH](https://data.mendeley.com/datasets/8kt969dhx6/1))
3. NIH chest X-rays ([NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data))


## Training:
Training procedures and details can be found in main.py.
Our trained chest X-ray model is provided [here](https://drive.google.com/file/d/1svxfab5YG2BVoSKe99krhwWeqQyQFUqw/view?usp=drive_link).


## Prototype visualisation:
Prototypes are visualized as their nearest training image patches.
HierProtoPNet generates semantically-dissimilar prototypes at different hierarchical levels due to the prototype mining paradigm: high-level prototypes focus on the most salient cancerous areas, the mid-level prototypes localise the difficult (i.e., less conspicuous) cancer-boundary areas, and the low-level prototypes capture sparser and finer cancer regions.

<div align=center>
<img width="900" height="315" src="https://github.com/cwangrun/HierProtoPNet/blob/master/img/prototypes.png"/></dev>
</div>



## Lesion Localisation:
Breast cancer:
<div align=center>
<img width="800" height="420" src="https://github.com/cwangrun/HierProtoPNet/blob/master/img/mammo.png"/></dev>
</div>

 
Thoracic disease:
<div align=center>
<img width="600" height="500" src="https://github.com/cwangrun/HierProtoPNet/blob/master/img/chestxray.png"/></dev>
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

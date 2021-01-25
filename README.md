
# COVID-19_chest-xray_classifier

### Overview
The COVID-19 pandemic still having devastating effect on the health and well-being of the global population. A critical step in the fight against COVID-19 is effective screening of infected patients, with one of the key screening approaches being radiology examination using chest radiography. Deep-learning artificial intelligent (AI) methods have the potential to help improve diagnostic efficiency and accuracy for reading portable CXRs.



### Implementation details
| Feature | Brief Explanation |
| ------ | ------ |
| Base Model Architecture | Resnet18 from [ResNet](https://arxiv.org/abs/1512.03385) family, implemetation from [PyTorch](https://pytorch.org/)|
| Learning Rate Finder | [Learning rate finder](https://arxiv.org/abs/1506.01186) implemetation from [FastAI](https://www.fast.ai/) |
| Learning rate and  Momentum scheduler| [One cycle policy](https://arxiv.org/abs/1803.09820) implemetation from [FastAI](https://www.fast.ai/)  to achieve superconvergence |
| Explainability | Implemented [gradcam](https://arxiv.org/abs/1610.02391) |
| Dataset |  Dataset [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) from [Tawsifur Rahman](https://www.kaggle.com/tawsifurrahman/datasets) |


### Results
| Model | Metrics(Accuracy) | Epochs |
| ------ | ------ | ------ |
| Resnet18 | 96 | 8 |

#### Confusion Matrix

#### Inference

## License
This project is licensed under the [MIT License](https://github.com/gurucharanmk/COVID-19_chest-xray_classifier/blob/main/LICENSE)

# WaveMultiCast
Official implementation of "[WaveMultiCast: Precipitation Nowcasting with Wavelet Transform and Multi-Branch Core Layeyr"

## Code

### Environment

```shell
conda env create -f env.yaml
conda activate wavemulticast
```
<details close>
<summary>Optional Accelerate Env</summary>

 We apply the `HuggingFace Accelerator` in our code to utilize multi-gpus. 
 One can config the accelerator env before runing code.

-  config the accelerate: `accelerate config`      
- apply accelerate to run code: `accelerate launch *.py`
</details>

### Datasets
All the two datasets in our paper is publicly available.
You can find the datasets as follows:
- [SEVIR](https://nbviewer.org/github/MIT-AI-Accelerator/eie-sevir/blob/master/examples/SEVIR_Tutorial.ipynb)
- [Shanghai_Radar](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2GKMQJ)

### Evaluation
```shell
# Note: Config the dataset path in `dataset/get_dataset.py` before running.
python run.py --backbone wavemulticast --eval --ckpt_milestone resources/your_checkpoint.pt  
```
### Backbone Training
```shell
python run.py --backbone wavemulticast
```
You can check the experimental configuration by
```shell
python run.py -h
```

## Acknowledgement

We refer to implementations of the following repositories and sincerely thank their contribution for the community:
- [Diffcast](https://github.com/DeminYu98/DiffCast/blob/main/README.md)
- [OpenSTL](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/README.md)

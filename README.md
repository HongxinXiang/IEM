# IEM

Official PyTorch-based implementation of Paper "A Image-enhanced Molecular Graph Representation Learning Framework".

[[Paper](#)] [[Appendix](https://github.com/HongxinXiang/IEM/blob/main/assets/appendix.pdf)]



## News!

**[2024/04/17]** Accepted in IJCAI 2024 !!!

**[2024/01/17]** Repository installation completed.



## TODO

- [x] Publish pre-training dataset
- [x] Publish supplementary material of IEM
- [x] Publish downstream task data
- [x] Release pre-trained teacher model



## Environments

#### 1. GPU environment

CUDA 11.6

Ubuntu 18.04



#### 2. create conda environment

```bash
# create conda env
conda create -n IEM python=3.9
conda activate IEM

# install environment
pip install rdkit
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install timm==0.6.12
pip install tensorboard
pip install scikit-learn
pip install setuptools==59.5.0
pip install pandas
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
pip install torch-geometric==1.6.0
pip install dgl-cu116
pip install ogb
```



## Pre-Training Teacher Model

The pre-trained teacher model and pre-trained datasets can be accessed in following table.

| Name                | Download link                                                | Description                                                  |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Pre-trained teacher | [IEM.pth](https://1drv.ms/u/s!Atau0ecyBQNTb0DCbVjgADxvcwo?e=580vg5) | You can download the teacher and put it in the directory: `resumes/pretrained-teachers/`. |
| Pre-trained dataset | [iem-200w](https://1drv.ms/f/s!Atau0ecyBQNTgRA-I02I_ED7s93u?e=lnmHND) | If you want to pre-train your own teacher model, please download the dataset and put it in `datasets/pre-training/iem-200w/processed/` |

If you want to pre-train your own teacher model, see the command below.



Usage:

```python
usage: pretrain_teacher.py [-h] [--dataroot DATAROOT] [--dataset DATASET]
                           [--workers WORKERS] [--nodes NODES]
                           [--ngpus_per_node NGPUS_PER_NODE]
                           [--dist-url DIST_URL] [--node_rank NODE_RANK]
                           [--model_name MODEL_NAME]
                           [--warmup_rate WARMUP_RATE] [--lr LR]
                           [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
                           [--weighted_loss] [--runseed RUNSEED]
                           [--start_epoch START_EPOCH] [--epochs EPOCHS]
                           [--batch BATCH] [--imageSize IMAGESIZE]
                           [--temperature TEMPERATURE]
                           [--base_temperature BASE_TEMPERATURE]
                           [--resume RESUME] [--n_ckpt_save N_CKPT_SAVE]
                           [--log_dir LOG_DIR]
```



run command to pre-train teacher:

```python
python pretrain_teacher.py \
	--nodes 1 \
	--ngpus_per_node 1 \
	--weighted_loss \
	--epochs 300 \
	--batch 128 \
	--lr 1e-2 \
	--ngpu 1 \
	--workers 4 \
	--dataroot ../datasets/pre-training/ \
	--dataset toy \
	--log_dir ./experiments/pretrain_teacher
```



## Distillation Training on Downstream Tasks

All downstream task data is publicly accessible below:

**8 classification tasks:**

| Datasets | #Molecules | #Task | Links                                                        |
| -------- | ---------- | ----- | ------------------------------------------------------------ |
| Tox21    | 7,831      | 12    | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgQyEq7V7amXDi7yn?e=O58EfS)] |
| ToxCast  | 8,576      | 617   | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgQ1QvgowlIH3Y0RP?e=t9kWcH)] |
| Sider    | 1,427      | 27    | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgQuYE9N7W_CRIRPE?e=LUXCdB)] |
| ClinTox  | 1,478      | 2     | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgQjeOeNgTJMhF5jz?e=0sPvnm)] |
| MUV      | 93,087     | 17    | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgQ8Evh0Vg9IjLF45?e=dwTE1X)] |
| HIV      | 41,127     | 1     | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgQ5Oq-42YJQ9kRm9?e=YRitpW)] |
| BBBP     | 2,039      | 1     | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgQrPNjO167wjZO6J?e=2Yie82)] |
| BACE     | 1,513      | 1     | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgQm-3Aqvp0HV0rX5?e=Zg6ILf)] |



**4 regression tasks:**

| Datasets | #Molecules | #Task | Links                                                        |
| -------- | ---------- | ----- | ------------------------------------------------------------ |
| ESOL     | 1,128      | 1     | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTfUrC5CvMJ2QP144?e=9UNSBc)] |
| Lipo     | 4,200      | 1     | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTfpFVfyj7t3WLXQc?e=fBWcUs)] |
| Malaria  | 9,999      | 1     | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTf8XsRVxMVmsK0bE?e=6XYEKM)] |
| CEP      | 29,978     | 1     | [[OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgQA4E_-P5aJ_9oPV?e=ykjZ9c)] |



Usage:

```python
usage: distillation_training.py [-h] [--dataroot DATAROOT] [--dataset DATASET]
                                [--graph_feat {min,all}]
                                [--label_column_name LABEL_COLUMN_NAME]
                                [--image_dir_name IMAGE_DIR_NAME] [--gpu GPU]
                                [--ngpu NGPU] [--workers WORKERS]
                                [--num_layers NUM_LAYERS]
                                [--feat_dim FEAT_DIM]
                                [--JK {concat,last,max,sum}]
                                [--t_dropout T_DROPOUT]
                                [--gnn_type {gin,gcn,gat,graphsage}]
                                [--resume_teacher RESUME_TEACHER]
                                [--resume_teacher_name RESUME_TEACHER_NAME]
                                [--lr LR] [--weight_t WEIGHT_T]
                                [--weight_kd WEIGHT_KD]
                                [--weight_ke WEIGHT_KE] [--seed SEED]
                                [--runseed RUNSEED] [--split {scaffold}]
                                [--epochs EPOCHS] [--start_epoch START_EPOCH]
                                [--batch BATCH] [--resume RESUME]
                                [--pretrain_gnn_path PRETRAIN_GNN_PATH]
                                [--model_name MODEL_NAME]
                                [--task_type {classification,regression}]
                                [--save_finetune_ckpt {0,1}]
                                [--log_dir LOG_DIR]
```



For examples, you can run the following code to distillation training:

```python
python distillation_training.py \
	--resume_teacher ../resumes/pretrained-teachers/image3d_teacher.pth \
	--resume_teacher_name image3d_teacher \
	--image_dir_name image3d \
    --dataroot ../datasets/fine-tuning/ \
	--dataset esol \
	--task_type regression \
	--feat_dim 512 \
	--batch 32 \
	--epochs 100 \
	--lr 0.001 \
	--split scaffold \
	--weight_kd 0.01 \
	--weight_ke 0.001 \
	--log_dir ./experiments/image3d_teacher/esol/rs0/ \
	--runseed 0 \
	--pretrain_gnn_path ../resumes/pretrained-gnns/GraphMVP.pth
```



# Reference


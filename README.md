# Breaking Isolation: Multimodal Graph Fusion for Multimedia Recommendation by Edge-wise Modulation

Pytorch implementation for the paper:
[Breaking Isolation: Multimodal Graph Fusion for Multimedia Recommendation by Edge-wise Modulation](https://dl.acm.org/doi/10.1145/3503161.3548399), ACM Mulimedia 2022.


## Environment Requirement
The code has been tested running under Python 3.8.5. The required packages are as follows:
* Pytorch 1.7.1
* CUDA 11.3
* torch-geometric 1.7.2

## Data
Download data following the instructions [here](https://github.com/weiyinwei/MMGCN#dataset).


## Training examples
* Movielens dataset (EgoGCN-hard)

`python main.py --l_r=0.0002 --weight_decay=0.0001 --weight_mode=GCN --num_routing=3 --data_path=movielens --batch_size 2048 --alpha_threshold='>0.4' --reattn=True --central=central_item`

* Tiktok dataset (EgoGCN-hard)

`python main.py --l_r=0.0003 --weight_decay=0.001 --weight_mode=GCN --num_routing=1 --data_path=Tiktok --batch_size 1024 --alpha_threshold '>0.6' --reattn=True --central=central_item`

For EgoGCN-soft, simply setting `--alpha_threshold '=0'`

## Other variants

For other variants in the paper, for instance, EgoGCN-NA,

`python main.py --l_r=0.0002 --weight_decay=0.0001 --weight_mode=GCN --num_routing=3 --data_path=movielens --batch_size 2048 --alpha_threshold='>99' --reattn=True --central=central_item --NA`

`python main.py --l_r=0.0003 --weight_decay=0.001 --weight_mode=GCN --num_routing=1 --data_path=Tiktok --batch_size 1024 --alpha_threshold '>99' --reattn=True --central=central_item --NA`

## Reference

If you found this code useful, please cite the following paper:
```
@inproceedings{DBLP:conf/mm/ChenWWZS22,
  author    = {Feiyu Chen and
               Junjie Wang and
               Yinwei Wei and
               Hai{-}Tao Zheng and
               Jie Shao},
  title     = {Breaking Isolation: Multimodal Graph Fusion for Multimedia Recommendation
               by Edge-wise Modulation},
  booktitle = {{MM} '22: The 30th {ACM} International Conference on Multimedia, Lisboa,
               Portugal, October 10 - 14, 2022},
  pages     = {385--394},
  publisher = {{ACM}},
  year      = {2022},
}
```

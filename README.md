# PST
Source code for the paper **PST: Measuring Skill Proficiency in Programming Exercise Process via Programming Skill Tracing**.

## Usage
### Download pre-trained CIG, CTG embedding
- [Google drive](https://drive.google.com/drive/folders/1PAyaS2xalpYrOzCpaHtfBiVaGQK3x_bJ?usp=share_link)
- Put the downloaded embedding folder into the root directory of the PST project.

### Train and test
```python
python train.py --dataset_name atcoder_c --num_exercises 1671 --do_test True
python train.py --dataset_name aizu_cpp --num_exercises 2207 --do_test True
```
### Train only
```python
python train.py --dataset_name atcoder_c --num_exercises 1671
python train.py --dataset_name aizu_cpp --num_exercises 2207
```
### Test only
```python
python test.py --dataset_name atcoder_c
python test.py --dataset_name aizu_cpp
```
- We removed some data that could not be processed by some baselines, such as submissions without corresponding exercise text. So the actual number of submissions of Atcoder_C for the experiment was 423841, and the actual number of submissions of AIZU_Cpp for the experiment was 264839.
## Corrections
1. In our paper, the equation for the cross-entropy loss function was written incorrectly, so the correct loss function for the PST model is as follows:
![PST_equation](/equation/PST.png)
2. We made a mistake in calculating the AUC of task 1 for all the baselines and PST. The good thing is that this mistake occurred in the final testing phase and did not affect model training, model selection, and the calculation of other metrics for task 1 and all metrics for other tasks. The correct experimental results of the PST model are as follows:
- Atcoder_C
  - Task1
    - AUC 0.8383
    - ACC 0.8107
  - Task2
    - RMSE 0.2875
  - Task3
    - RMSE 0.3453
  - Task4
    - RMSE 0.2862
- AIZU_Cpp
  - Task1
    - AUC 0.8849
    - ACC 0.9596
  - Task2
    - RMSE 0.2239
  - Task3
    - RMSE 0.3073
  - Task4
    - RMSE 0.1731

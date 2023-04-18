# PST
Source code for the paper **PST: Measuring Skill Proficiency in Programming Exercise Process via Programming Skill Tracing**.

## Usage
### Train and test
```python
train.py --dataset_name atcoder_c --num_exercises 1671 --do_test True
train.py --dataset_name aizu_cpp --num_exercises 2207 --do_test True
```
### Train only
```python
train.py --dataset_name atcoder_c --num_exercises 1671
train.py --dataset_name aizu_cpp --num_exercises 2207
```
### Test only
```python
test.py --dataset_name atcoder_c
test.py --dataset_name aizu_cpp
```
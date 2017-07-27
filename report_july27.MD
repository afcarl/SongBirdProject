All the results are on 120*8 original dimension <br/>
Currently the architecture is:<br/>
120\*8 -> 90\*8 -> 60\*8 -> 30\*8 -> 15\*8 (latent) (both mean and var)<br/>
-> 30\*8 -> 60\*8 -> 90\*8<br/>
-> 120\*8 (mean)<br/>
-> 1 (var)<br/>

mse loss

Running for 10 epochs.

## Initial results with all layers having tanh and 'rmsprop':

### Loss:
![image](https://user-images.githubusercontent.com/9252491/28666264-4a4941ea-72e4-11e7-83c6-0032d5ddbda2.png)
<br/>
### Loss from 1st iteration:
![image](https://user-images.githubusercontent.com/9252491/28666301-6f0ea3d0-72e4-11e7-84ad-de8853196073.png)
### Mean 
![image](https://user-images.githubusercontent.com/9252491/28666421-e3bd044c-72e4-11e7-8026-fc00531a53ce.png)
### Mean + var
![image](https://user-images.githubusercontent.com/9252491/28666394-c1878d84-72e4-11e7-91b0-3f3d7ffa011b.png)
### var
```
>>> var
array([[-3.53699708],
       [-3.63122916],
       [-3.57786202],
       [-3.57982898],
       [-3.62257671],
       [-3.55158973],
       [-3.56943727],
       [-3.52750468],
       [-3.58194995],
       [-3.53988194]], dtype=float32)

>>> np.exp(var/2)*np.random.normal(size=(samples, original_dim), loc=0.,scale=epsilon_std)
array([[ 0.37117556, -0.01878777, -0.0294915 , ..., -0.11070047,
        -0.11144824, -0.0724738 ],
       [-0.03779677,  0.04233758, -0.11830928, ...,  0.08563999,
        -0.05889469,  0.00590314],
       [ 0.13640007,  0.08482936,  0.06695945, ...,  0.33502804,
        -0.02712832,  0.25557621],
       ..., 
       [-0.0206035 ,  0.02031483,  0.19582266, ...,  0.25889071,
        -0.13886413, -0.16346034],
       [-0.28630196,  0.07093736, -0.05176849, ..., -0.06810686,
        -0.05720696, -0.28261106],
       [-0.07728023, -0.01441295, -0.05702819, ..., -0.1203734 ,
        -0.15280869, -0.28172838]])

```

## Results with all tanh and adam as optimizer

### Loss
![image](https://user-images.githubusercontent.com/9252491/28666731-4032b0ea-72e6-11e7-8a79-80c83425696a.png)

### Loss from 1st iter
![image](https://user-images.githubusercontent.com/9252491/28666774-6a1b277a-72e6-11e7-8203-ae83431c83ba.png)

### Mean
![image](https://user-images.githubusercontent.com/9252491/28666809-9034400e-72e6-11e7-821d-0476784dc259.png)

### Mean + Var
![image](https://user-images.githubusercontent.com/9252491/28666839-a5a7cd02-72e6-11e7-9519-c9ffbbb259b2.png)

### Var

```
>>> var
array([[-3.61829615],
       [-3.6220963 ],
       [-3.61824393],
       [-3.61828065],
       [-3.63098025],
       [-3.61828637],
       [-3.61812949],
       [-3.61830115],
       [-3.6308496 ],
       [-3.63080096]], dtype=float32)

>>> print (np.exp(var/2)*np.random.normal(size=(samples, original_dim), loc=0.,scale=epsilon_std))
[[ 0.3321669  -0.03561313 -0.13075393 ...,  0.16846737 -0.08728551
  -0.0115059 ]
 [ 0.23305575  0.21126566 -0.12404597 ..., -0.31786065  0.14101535
   0.04280035]
 [-0.26009264  0.16150949  0.0785434  ..., -0.11492866  0.01787932
  -0.05043118]
 ..., 
 [-0.07089747  0.1173939  -0.04739465 ..., -0.17757838 -0.10433422
   0.11735423]
 [-0.4490804   0.0007472   0.08967688 ..., -0.2004654  -0.302117
  -0.00063129]
 [-0.01945704 -0.19417686  0.08256923 ...,  0.1830672  -0.34806057
  -0.02569835]]
```

## Results with all leaky relu and adam as optimizer

### Loss
![image](https://user-images.githubusercontent.com/9252491/28667800-aa755b2a-72ea-11e7-9c6f-4fc3e85b5266.png)

### Loss from 1st iter
![image](https://user-images.githubusercontent.com/9252491/28667834-d06ad896-72ea-11e7-9ee1-cd3d0af95f82.png)

### Mean
![image](https://user-images.githubusercontent.com/9252491/28667866-f84c38c8-72ea-11e7-8b71-a03c4678978e.png)

### Mean + Var
![image](https://user-images.githubusercontent.com/9252491/28667878-0a6223d8-72eb-11e7-8dd3-e41b6fdf529a.png)

### Var
```
>>> print (var)
[[-3.56234121]
 [-3.5790987 ]
 [-3.52526832]
 [-3.56495714]
 [-3.60306311]
 [-3.52681589]
 [-3.51729107]
 [-3.50151682]
 [-3.52240658]
 [-3.63412189]]

>>> print (np.exp(var/2)*np.random.normal(size=(samples, original_dim), loc=0.,scale=epsilon_std))
[[-0.12287621  0.25512763  0.00804883 ..., -0.06400764 -0.17941274
  -0.08879864]
 [-0.327887    0.15359486 -0.10946708 ..., -0.13625387  0.21510993
   0.08784759]
 [ 0.11402627 -0.10499514 -0.09132854 ..., -0.04476571  0.32059429
  -0.17712243]
 ..., 
 [ 0.03260066  0.00600564 -0.36579702 ...,  0.03515057  0.04487984
  -0.07943941]
 [ 0.07878341 -0.13492589  0.00361673 ...,  0.10670318  0.12669274
  -0.0884982 ]
 [ 0.11078715  0.08144164 -0.13095651 ...,  0.1854084   0.18917225
  -0.10228315]]
```

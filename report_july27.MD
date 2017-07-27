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


# MuSeRC


Version1:

Train: 
```
	Texts:  500
	Questions:  2896
	Answers:  11949
```

Val: 

```
	Texts:  100
	Questions:  529
	Answers:  2235
```

Test: 
```
	Texts:  322
	Questions:  1812
	Answers:  7613
```

Overlap 3-5

 
Metrics:

Results on 1378 examples (no confused sentences, no sentences with bad annotators):

```
Per question measures (i.e. precision-recall per question, then average) 
	P: 0.9637155297532656 - R: 0.701983551040155 - F1m: 0.8122865138960143


EM: 0.42


Dataset-wide measures (i.e. precision-recall across all the candidate-answers in the dataset) 
	P: 0.9611111111111111 - R: 0.6942721634439986 - F1a: 0.8061851302690108
```
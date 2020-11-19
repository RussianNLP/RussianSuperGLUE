# RussianSuperGLUE

### Russian SuperGLUE benchmark

We introduce an advanced Russian  general  language  understanding  evaluation benchmark.

Recent advances in the field of universal language models and transformers require the development of a methodology for their broad diagnostics  and  testing  for  general  intellectual skills  -  detection  of  natural  language  inference,  commonsense reasoning,  ability to perform  simple  logical  operations  regardless  of text  subject  or  lexicon.   For  the  first  time,  a benchmark of nine tasks,  collected and organized analogically to the SuperGLUE methodology, was developed from scratch for the Russian language.  We provide baselines,  human  level  evaluation,  an  open-source  framework  for  evaluating  models and an overall leaderboard of transformer models for the Russian language.

## RELEASE v1.1

- update and expand some datasets:
    - RUSSE: new test + human benchmark
    - DaNetQA: expand the dataset + new test + human benchmark
    - RuCoS: expand the dataset + clean typos/inaccuracies
    - MuSeRC: expand the dataset + clean typos/inaccuracies
- add and improve code for jiant:
    - evaluation of models: GPT-3, GPT-2
    - correct lidirus preprocessing
- fix typos and bugs
- refactor web interface and improved reliability of the model evaluation system on the website

## Instructions:

[Jupyter link](https://github.com/RussianNLP/RussianSuperGLUE/blob/master/Russian_SuperGLUE_example.ipynb)

## Leaderboard:

[Russiansuperglue.com](https://russiansuperglue.com/)

## Download the Data:

[All the tasks (zip)](https://russiansuperglue.com/tasks/download)

[Some tasks from the website](https://russiansuperglue.com/tasks/)

## Documentation:

You can see our documentation at [diagnostics description](https://russiansuperglue.com/datasets/)

 - LiDiRus [link](https://russiansuperglue.com/tasks/task_info/LiDiRus)
 - RCB [link](https://russiansuperglue.com/tasks/task_info/RCB)
 - PARus [link](https://russiansuperglue.com/tasks/task_info/PARus)
 - MuSeRC [link](https://russiansuperglue.com/tasks/task_info/MuSeRC)
 - TERRa [link](https://russiansuperglue.com/tasks/task_info/TERRa)
 - RUSSE [link](https://russiansuperglue.com/tasks/task_info/RUSSE)
 - RWSD [link](https://russiansuperglue.com/tasks/task_info/RWSD)
 - DaNetQA [link](https://russiansuperglue.com/tasks/task_info/DaNetQA)
 - RuCoS [link](https://russiansuperglue.com/tasks/task_info/RuCoS)

## Cite us:

Read our [article](https://www.aclweb.org/anthology/2020.emnlp-main.381.pdf)

Please, cite us this way:
```
@article{shavrina2020russiansuperglue,
                  title={RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark},
                  author={Shavrina, Tatiana and Fenogenova, Alena and Emelyanov, Anton and Shevelev, Denis and Artemova, Ekaterina and Malykh, Valentin and Mikhailov, Vladislav and Tikhonova, Maria and Chertok, Andrey and Evlampiev, Andrey},
                  journal={arXiv preprint arXiv:2010.15925},
                  year={2020}
                  }
```

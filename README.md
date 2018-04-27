# Layered Memory Network (LMN)
The LMN model ranked 1st place on [MovieQA Video+Subtt-based Answering Challenge 2017](http://movieqa.cs.toronto.edu/workshops/iccv2017/) ([The Joint Video and Language Understanding Workshop, ICCV 2017](https://sites.google.com/site/describingmovies/workshop-at-iccv-17)).

- The flowchart of Layered Memory Network (LMN).

![LMN](https://raw.githubusercontent.com/bowong/Layered-Memory-Network/master/img/framework.jpg)

- The framework of Dynamic Subtitle Memory module with update mechanism.

![DSM](https://raw.githubusercontent.com/bowong/Layered-Memory-Network/master/img/dynamic.jpg)



## Train


```
python mqa_video+subtitle+update+question.py
```

## Paper

Bo Wang, Youjiang Xu, Yahong Han, Richang Hong. ["Movie Question Answering: Remembering the Textual Cues for Layered Visual Contents."](https://arxiv.org/abs/1804.09412) AAAI, 2018. [[Paper]](https://arxiv.org/abs/1804.09412)
```
@inproceedings{Wang2018,
  author    = {Bo Wang and Youjiang Xu and Yahong Han and Richang Hong},
  title     = {Movie Question Answering: Remembering the Textual Cues for Layered Visual Contents},
  booktitle = {AAAI},
  year      = {2018},
}
```


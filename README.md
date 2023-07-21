# CertPri
We propose CertPri, a test input prioritization technique designed based on a movement cost perspective of test inputs in DNNs’ feature space. 
CertPri differs from previous works in three key aspects:
(1) certifiable - it provides formal robustness guarantee for the movement cost; 
(2) effective - it leverages formal guaranteed movement costs to identify malicious bug-revealing test inputs; and 
(3) generic - it can be applied to various tasks, data forms, models, and scenarios. 
CertPri significantly improves 53.97% prioritization effectiveness on average compared with baselines. 
Besides, its robustness and generalizability are 1.41-2.00 times and 1.33-3.39 times that of baselines on average, respectively.

## Citing CertPri

A technical description of AI Fairness 360 is available in this
[paper](http://arxiv.org/abs/2307.09375). Below is the bibtex entry for this paper.

```
@inproceedings{Zheng2023CertPri,
   author = {Zheng, Haibin and Chen, Jinyin and Jin, Haibo},
   title = {CertPri: Certifiable Prioritization for Deep Neural Networks via Movement Cost in Feature Space},
   booktitle = {38th IEEE/ACM International Conference on Automated Software Engineering},
   address = {Belval, Esch-sur-Alzette, Luxembourg},
   pages = {1-13},
   date = {September 11 - 15},
   publisher = {{IEEE/ACM}},
   year = {2023}
}
```

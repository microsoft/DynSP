
# DynSP (Dynamic Neural Semantic Parser)

This project contains the source code of the Dynamic Neural Semantic Parser (DynSP), 
based on [DyNet](https://github.com/clab/dynet). 

Detail of DynSP can be found in the following ACL-2017 paper:

[Mohit Iyyer](https://people.cs.umass.edu/~miyyer/), [Wen-tau Yih](http://scottyih.org), [Ming-Wei Chang](https://ming-wei-chang.github.io/).
[Search-based Neural Structured Learning for Sequential Question Answering.](http://aclweb.org/anthology/P17-1167) ACL-2017.

    @InProceedings{iyyer-yih-chang:2017:Long,
      author    = {Iyyer, Mohit  and  Yih, Wen-tau  and  Chang, Ming-Wei},
      title     = {Search-based Neural Structured Learning for Sequential Question Answering},
      booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      month     = {July},
      year      = {2017},
      address   = {Vancouver, Canada},
      publisher = {Association for Computational Linguistics},
      pages     = {1821--1831},
    }

The output files and the models for producing the reported results are also included.  Below are the scripts that
produces the results reported in Table 2 of the paper (DynSP and DynSP*).

```bash
$ ./check.sh moduleKwNoMap-b15-ind
moduleKwNoMap-b15-ind
Best Accuracy: 0.425963 (Reward: 0.479099) at epoch 28
Best Accuracy: 0.369095 (Reward: 0.423323) at epoch 10
Best Accuracy: 0.348668 (Reward: 0.405460) at epoch 24
Best Accuracy: 0.377477 (Reward: 0.439594) at epoch 29
Best Accuracy: 0.349951 (Reward: 0.413719) at epoch 20
Best Accuracy: 0.351802 (Reward: 0.409400) at epoch 19
0.359399 20
```

```bash
$ ./evalModel-indep.sh moduleKwNoMap-b15-ind 20
Sequence Accuracy = 10.15% (104/1025)
Answer Accuracy =   41.97% (1264/3012)
Break-down:
Position 0 Accuracy = 70.93% (727/1025)
Position 1 Accuracy = 35.84% (367/1024)
Position 2 Accuracy = 20.06% (137/683)
Position 3 Accuracy = 12.23% (28/229)
Position 4 Accuracy = 13.16% (5/38)
Position 5 Accuracy = 0.00% (0/9)
Position 6 Accuracy = 0.00% (0/4)
```

```bash
$ ./check.sh moduleKwNoMap-b15
moduleKwNoMap-b15
Best Accuracy: 0.450863 (Reward: 0.516281) at epoch 17
Best Accuracy: 0.379691 (Reward: 0.439837) at epoch 12
Best Accuracy: 0.366021 (Reward: 0.422335) at epoch 16
Best Accuracy: 0.391892 (Reward: 0.456894) at epoch 26
Best Accuracy: 0.370968 (Reward: 0.442918) at epoch 20
Best Accuracy: 0.368468 (Reward: 0.431721) at epoch 18
0.375408 18
```

```bash
$ ./evalModel.sh moduleKwNoMap-b15 18
Sequence Accuracy = 12.78% (131/1025)
Answer Accuracy =   44.65% (1345/3012)
Break-down:
Position 0 Accuracy = 70.44% (722/1025)
Position 1 Accuracy = 41.11% (421/1024)
Position 2 Accuracy = 23.57% (161/683)
Position 3 Accuracy = 13.97% (32/229)
Position 4 Accuracy = 18.42% (7/38)
Position 5 Accuracy = 11.11% (1/9)
Position 6 Accuracy = 25.00% (1/4)
```    

_A [tokenzier](https://github.com/myleott/ark-twokenize-py) and some data files are not included in the initial release 
due to licencing issues. They can be found at a [fork](https://github.com/scottyih/DynSP)._ 
The [Sequential Question Answering (SQA) dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54253), 
published and used in the same paper, can be downloaded separately.




# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

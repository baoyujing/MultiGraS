## Install ROUGE-1.5.5
1. Download ROUGE-1.5.5\
```<https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5>```\
```export ROUGE_EVAL_HOME="/absolute_path_to/ROUGE-1.5.5/data/"```

2. Install Perl Packages\
```sudo apt-get install perl```\
```sudo apt-get update```\
```sudo cpan install XML::DOM```

3. Remove files to avoid ERROR of the .db files:\
```rm WordNet-2.0.exc.db```\
```./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db```

4. Install ```pyrouge```\
```pip install pyrouge```\
```pyrouge_set_rouge_path /absolute_path_to/ROUGE-1.5.5/```

References:
1. <https://medium.com/@sagor_sarker/how-to-install-rouge-pyrouge-in-ubuntu-16-04-7f0ec1cda81b>
2. <https://blog.csdn.net/MerryCao/article/details/49174283>
3. <https://github.com/bheinzerling/pyrouge>
4. <https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5>
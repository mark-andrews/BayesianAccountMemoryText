This is the RMarkdown+LaTeX source for the manuscript entitled *A Bayesian
Account of Memory for Text* by Mark Andrews.

The pdf can be compiled with the following command:
```
Rscript -e "rmarkdown::render('main.Rmd')"
```

First, get the BibTeX library `mjandrews.bib` from <https://github.com/mark-andrews/bibtex>. For example, you can use `wget` as follows:
```
wget https://raw.githubusercontent.com/mark-andrews/bibtex/master/mjandrews.bib -O mjandrews.bib
```

The `Makefile` contains these commands for your convenience.

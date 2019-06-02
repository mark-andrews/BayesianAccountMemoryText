pdf:
	Rscript -e "rmarkdown::render('main.Rmd')"

clean:
	rm -f main.log main.tex 
	rm -f child_rmd_files/open_science_stuff.tex child_rmd_files/out-of-sample.tex child_rmd_files/sample_size_determination.tex 
	rm -f child_rmd_files/main-tikzDictionary child_rmd_files/text_*tex
	rm -rf child_rmd_files/tmp

get_bibtex:
	wget https://raw.githubusercontent.com/mark-andrews/bibtex/master/mjandrews.bib -O mjandrews.bib


# Modelling with Gustav

The following is the usage notes for `gustav`: 

```bash
Gustav: Probabilistic Topic Modelling Toolbox

Usage:
  gustave model new [--model-type=<model_type>] <corpus_name> [--K_min=<K_min>] [--K_max=<K_max>]
  gustave model <model_name> update [--iterations=<N>] [--hyperparameters] [--parallel=<K>]
  gustave data new <corpus_name> [--data-type=<data_type>] <text_file> <vocab_file>
  gustave init 
  gustave (-h | --help)
  gustave --version

Options:
  -h --help                     Show this screen.
  --version                     Show version.
  --parallel=<K>                Number of processors [default: 1]
  --iterations=<N>              Model update iterations [default: 100]
  --model-type=<model_type>     Type of topic model [default: hdptm].
  --data-type=<data_type>       Type of data set [default: bag_of_words].
  --K_min=<K_min>               Minimum number of topics [default: 10]
  --K_max=<K_max>               Maximum number of topics [default: 100]
```
## Example: Create a corpus

```bash
gustave data new foo example_corpus.txt vocab.txt
```
where `example_corpus.txt` is a text corpus where the "texts" are delimited by
line breaks and the "words" are delimited by "|", e.g. 
```bash
foo|bar|foobar|foo|foo
foobar|foo|bar|bar|bar
bar|foo|bar|foo|bar
```
and `vocab.txt` is a line break delimited list of word types, e.g. 
```bash
foo
bar
foobar
```

## Example: Initialize your topic model using corpus `foo`

```bash
gustave model new foo --K_min=1000 --K_max=2500
```

The created model will be given a random name like `hdptm_180117202450_6333` where the first string of digits is datetimestamp and the second is random integer.

## Example: Update the topic model

Update the model for 1000 iterations.
```bash
gustave model hdptm_180117202450_6333 update --parallel 16 --iterations=1000 --hyperparameters
```

## Saving results

Corpora and samples are saved inside a directory called `data`, and all details are stored in the config file `gustav.cfg`.

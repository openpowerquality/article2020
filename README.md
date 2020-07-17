# article2020

Repo for the article on the pilot study project and results of Ph.D. work, to be submitted to [Energies](https://www.mdpi.com/journal/energies).

## Creating the draft

Run pdflatex, bibtex, and pdflatex twice more to resolve all references and citations:
```
make all
```

Run just pdflatex (no new references or citations to resolve):

```
make
```

Get rid of intermediate files:

```
make clean
```


## Misc notes

* Power Quality Event Detection: https://www.mdpi.com/1996-1073/11/1/145

* https://en.wikipedia.org/wiki/Solar_power_in_Hawaii

* Remember to revert \documentclass before submission.

## Revision:

Introduction
  * Goals to Aims
  * Include "research gap", hypotheses

OPQ Architecture Section
  * Move it into methods

Related Work
  * keep the same

Methods
  * Build the hardware and software
  * Perform verification of hardware and software in a lab setting.
  * Perform a pilot study to validate the utility of the architecture in a real-world setting


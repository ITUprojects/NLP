# Language Processing and Deep Learning 2024<br><br>Exam Project: *Cross-Language Named Entity Recognition (NER) Performance on Pokémon Texts*

## Table of Contents

- [Group Members](#group-members)
- [Abstract](#abstract)
- [Folder Structure](#folder-structure)

---

## Group Members

- [TheColorman](https://github.com/TheColorman) - contact@colorman.me  
- [KayisWasTaken](https://github.com/KayisWasTaken) - serr@itu.dk  
- [m3bogdan](https://github.com/m3bogdan) - bomi@itu.dk  
- [Pheadar](https://github.com/Pheadar) - peca@itu.dk  

---

## Abstract

Using Pokémon anime synopses as the domain-specific text, this study assesses the performance of a multilingual BERT (mBERT) model on Named Entity Recognition (NER) tasks across many languages. Through synopsis extraction and annotation from TheTVDB, we investigate mBERT's named entity recognition capabilities in German, French, and English. We create a unique tag specifically for names connected to Pokémon and put in place a BIO tagging system. Our results show that the English-trained model performs the best overall. The French-trained model, however, continuously performs worse, indicating problems with the volume and quality of the data. Our findings show that domain-specific cross-language transfer is possible despite these obstacles, while more study is required to maximize model performance between languages.

---

## Folder Structure

```
.
├── Project
│   ├── baseline
│   ├── cache
│   ├── data
│   ├── plots
│   ├── project-desc
│   ├── ReferenceText
│   └── src
├── README.md
└── shell.nix
```

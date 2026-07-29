# CP468 Group 22 Final Project

# CP468 Group 22 Headline Generation Final Project

## Course
CP468-D — Artificial Intelligence

## Project
Sequence-to-Sequence Modeling: LSTM vs. LLM

## Task
Headline generation: news article body → headline.

## Group 22 Members
- Jack Hargrave
- Jayvyn Viggers
- Josh To
- Maxwell Posadas
- Averi Wylie

## Project Overview
This project compares a classical LSTM-based sequence-to-sequence model with attention against a modern large language model on the task of headline generation.

The LSTM model will be trained from scratch on a public news headline dataset. The LLM baseline will be tested on the same held-out test set using zero-shot and few-shot prompting.

## Repository Structure

```text
src/                  Source code
data/raw/             Raw dataset files or download cache
data/processed/       Processed train/validation/test CSV files
outputs/models/       Saved trained models
outputs/predictions/  Model predictions
outputs/metrics/      Evaluation results
notebooks/            Data inspection notebooks
report/figures/       Figures and tables for the report

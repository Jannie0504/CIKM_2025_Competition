# CIKM 2025 Competition: Multilingual E-commerce Product Search Competition: Multilingual Query-Category and Query-Item Relevance

## Run the Code

### Data

Download the data and put it in the folder .data/
```tree
.
├── data
│   ├── train_QC_en_clean_norm_up.txt
│   ├── train_QI_en_clean_norm_up.txt
│   ├── test_QC_en_clean_norm_up.txt
│   └── test_QI_en_clean_norm_up.txt
```

### Training

```shell
# Run the training
./scripts/train.sh

# Run the training for the category prediction (QC) task
./scripts/train_QC.sh

# Run the training for the query-item relevance (QI) task
./scripts/train_QI.sh
```

### Generate Submit File
After running the training code, the prediction results should be in the folder .predictions/

```tree
.
├── predictions
│   ├── submit_QC.txt
│   └── submit_QI.txt
```

gen_submit_txt.sh produces submit/submit_QC.txt and submit/submit_QI.txt

### Submit the results

Zip the two files in submit.zip. You can use the following command:
```shell
# Compress the prediction files
./scripts/compress_submit.sh
```

Then, upload submit.zip to the platform.

### Submit your code
If you need to submit your code, it should be submitted in a zip file, and reproducible.
- It should have a train.sh and a predict.sh files that enables to reproduce the training and prediction parts, in the script/ folder.
- Modifications of data, or new data, should be included in the folder data/
- A README should contain additional explanations on how to reproduce the results.
- Besides, a requirements.txt should be included
- The final model should be in the folder model/

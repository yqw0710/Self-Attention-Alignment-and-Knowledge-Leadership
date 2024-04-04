## Self-Attention-Alignment-and-Knowledge-Leadership

## Catalogue description

  * data

    + All data used in the experiment

  * model_save

    + Storing fine-tuned models

      

## 1、Configuration environment

```bash
pip install -r requirements.txt
```

## 2、Prepare data

  * data
    + alignment：Data on debiasing
    + viaual：Attention Visualization Related Data
      All data can be downloaded at: https://drive.google.com/drive/folders/15dt7CW0V1x5eFgBEg8ATrmw-ixdYVULm

## 3、Run

## 3.1 Debias

```bash
export MODEL_TYPE=bert
export MODEL_NAME_OR_PATH=bert-base-uncased
export OUTPUT_DIR=../model_save/bios/bert/original

python alignment_debias.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_type $MODEL_TYPE \
  --output_dir $OUTPUT_DIR
  --batch_size 32 \
  --lr 1e-5 \
  --num_epochs 1 \
```

## 3.2 Visualization

```bash
python attention_visualisation.py
```

## Code Acknowledgements

**1.Intrinsic Debiasing Method**

  * Context-Debias:https://github.com/kanekomasahiro/context-debias
  * Auto-Debias:https://github.com/Irenehere/Auto-Debias
  * AttenD:https://github.com/YacineGACI/AttenD
  * MABEL:https://github.com/princeton-nlp/MABEL

   

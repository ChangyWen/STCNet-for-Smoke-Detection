# STCNet: Spatio-Temporal Cross Network for Industrial Smoke Detection
## Overview
* This is the implementation of the model proposed in [[1]](#references).
* Please refer to [Caoyichao/STCNet](https://github.com/Caoyichao/STCNet) for the repo given by the author of [[1]](#references). Note that, by the time I shared my code, the authors of [[1]](#references) have not made their implementation available. I shared the code for research study only.
* The result (the F-score on testing dataset) I got tally with the one shown in [1], which indicates the correctness of the implementation of this repo.
* The data we used is given by [[2]](#references). I used and modified code of [CMU-CREATE-Lab/deep-smoke-machine](https://github.com/CMU-CREATE-Lab/deep-smoke-machine) for downloading and pre-processing data.

## Usage
* Download and pre-process data (videos with 320 by 320 resolutions)
  ```bash
  bash data_preprocess.sh 320
  ```
* Training and validating (e.g., using GPU 0). The tensorboard log, prediction outputs (on validation data), and the trained model of training and validating will be saved. 
    ```bash
    python main.py --gpu 0
    ```
* Validating
    ```bash
    python main.py --test --mode validation
    ```
* Testing
    ```bash
    python main.py --test --mode test
    ```

## Training results
* The trained model is saved in [trained_model](./trained_model/).
* The prediction outputs are saved in [pred_out](./pred_out/).
* Results on validation dataset: 
  * Accuracy 0.925699 - Precision 0.923586 - Recall 0.913358 - F-score 0.918443
* Results on testing dataset: 
  * Accuracy 0.917490 - Precision 0.900443 - Recall 0.876863 - F-score 0.888497

## References
* [1] Y.  Cao,  Q.  Tang,  X.  Lu,  F.  Li,  and  J.  Cao,  “STCNet: Spatio-Temporal Cross Network for Industrial Smoke Detection,”arXiv preprintarXiv:2011.04863, 2020.
* [2] Y.-C. Hsu, T.-H. K. Huang, T.-Y. Hu, P. Dille, S. Prendi, R. Hoffman, A. Tsuhlares, J. Pachuta, R. Sargent, and I. Nourbakhsh, “Project RISE: Recognizing Industrial Smoke Emissions,” in Proc. of AAAI, 2021.
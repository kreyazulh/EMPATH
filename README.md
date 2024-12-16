# EMPATH: MediaPipe-aided Ensemble Learning with Attention-based Transformers for Accurate Recognition of Bangla Word-Level Sign Language

Welcome to the official GitHub repository for EMPATH, a cutting-edge framework designed to significantly enhance the recognition of Bangla Word-Level Sign Language (BdSL). EMPATH integrates the robust capabilities of MediaPipe Holistic for landmark detection and tracking with the power of ensemble learning and attention-based Transformer models to set new benchmarks in sign language recognition accuracy.

## Overview

EMPATH is designed to address the challenges in accurately recognizing isolated sign language by leveraging a synergistic approach that combines advanced machine learning techniques. It showcases exceptional performance across various datasets, including BdSL-40, BDSign-Word, INCLUDE, WLASL, and MSL Medical Dataset demonstrating its adaptability to diverse linguistic contexts. Furthermore, EMPATH employs a specialized interpolation model to effectively handle missing keypoints, showcasing its robust capabilities and identifying areas for enhancement.

![MediaPipe Holistic](https://github.com/kreyazulh/EMPATH/assets/87698516/3153b5f4-560b-4269-80f2-7b3fafa37dcf)


## Key Features

- **MediaPipe Holistic Integration:** Utilizes advanced landmark extraction techniques for optimal detection and tracking, with configurable parameters to balance performance and inference time.
- **Ensemble Learning with Transformers:** Employs an ensemble of Transformer models to enhance prediction accuracy and reliability.
- **Interpolation Model for Missing Keypoints:** A specialized approach to address the challenge of missing data in sign language video frames, ensuring comprehensive analysis.
- **Extensive Dataset Evaluation:** Rigorously trained, tested, and validated across multiple datasets to highlight EMPATH's strengths and areas for improvement.
- **Adaptability to Diverse Linguistic Contexts:** Demonstrated success in recognizing a broad range of sign languages, with a focus on Bangla Word-Level Sign Language.

## Pre-trained Models and Preprocessed Files

We have made available pre-trained models and preprocessed files for the INCLUDE, INCLUDE-50, SignBD-Word-90, and MSL Medical Dataset for easy replication and extension of our work. You can download these resources from the following link:

[Download Pre-trained Models and Preprocessed Files](https://drive.google.com/drive/u/0/folders/1W80b38_ZmfkdbcO8iTyqb8JFXNxdLT9E)

These resources are intended to provide a quick start for working with these datasets. For the remaining datasets, similar preprocessing and training procedures as detailed in our documentation can be applied to achieve comparable results.


https://github.com/kreyazulh/EMPATH/assets/87698516/0e2a268d-7704-4123-b13e-c3f3a66a4eaf

## Dataset Links

For convenience, we have provided the dataset sources utilized in our EMPATH experiments. These datasets include BdSL-40, SignBD-Word, INCLUDE, WLASL, and the MSL Medical Dataset.

- **BdSL-40 Dataset**: [Download BdSL-40](https://github.com/Patchwork53/BdSL40_Dataset_AI_for_Bangla_2.0_Honorable_Mention/tree/main)
- **SignBD-Word Dataset**: [Download SignBD-Word](https://doi.org/10.5281/zenodo.6779843)
- **INCLUDE Dataset**: [Download INCLUDE](https://zenodo.org/records/4010759)
- **WLASL Dataset**: [Download WLASL](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)
- **MSL Medical Dataset**: [Download MSL Medical](https://www.kaggle.com/datasets/arkuuu21/msl-medical)
## Running the Code

### Python Scripts

All Python `.py` files in this repository are designed to be straightforward to execute. To run any script, simply use the following command in your terminal:

```bash
python3 <file.py>
```

### Notebooks

Notebooks can be run in any environment like jupyter notebook, kaggle, collab or local machine IDEs as preference.

## References

If you find this work helpful, please consider citing the following publications:

- @inproceedings{hasan2025empath,
  title={EMPATH: MediaPipe-Aided Ensemble Learning with Attention-Based Transformers for Accurate Recognition of Bangla Word-Level Sign Language},
  author={Hasan, Kazi Reyazul and Adnan, Muhammad Abdullah},
  booktitle={International Conference on Pattern Recognition},
  pages={355--371},
  year={2025},
  organization={Springer}
}


- @article{hasanimplementation,
  title={Implementation and Reproducibility Notes on EMPATH: Enhancing Word-Level Sign Language Recognition},
  author={Hasan, Kazi Reyazul and Abdullah, Muhammad}
}





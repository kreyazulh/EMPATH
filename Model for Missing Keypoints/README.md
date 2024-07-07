# Predicting Missing Hand Keypoints in Sign Language Tasks

## Key Features

- **Linear Interpolation Model:** Implementing a regression-based algorithm to address missing hand landmarks.
- **Wrist Keypoint Alignment:** Aligning hand wrist keypoints with pose wrist keypoints to rectify inconsistencies between arm and hand positions.
- **Memory Management:** Utilizing a garbage collector-like strategy to manage memory, releasing and writing missing values at designated intervals.
- **Optimized Inference:** Optimizing the algorithm to run only once for each frame, minimizing additional computation costs in terms of inference time.


   <p align="center">
  <img src="https://github.com/kreyazulh/EMPATH/assets/87698516/6e10dff9-aacb-4458-9865-4a27f8e72501" alt="Before" style="width:45%;" />
  <img src="https://github.com/kreyazulh/EMPATH/assets/87698516/b002bfad-fb31-40f1-9217-3c9fbd5b2a1e" alt="After" style="width:45%;" />
</p>

<p align="center">Before and after (reference video frame WLASL)</p>





## Dependencies

Ensure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```
## Trial Run

You can try the base MediaPipe Holistic code by using:

```python
python3 mediapipe_only.py boi.mp4
```
To run with the algorithm implementation, use:

```python
python3 fix_keypoints.py boi.mp4
```
The sample video was taken from [SignBD-Word](https://zenodo.org/records/6779843) dataset. Note that runtime of the codes may vary because of model complexity, video quality and size. You can experiment with other videos as desired.
To analyze result differences further, you can split your videos into frames using **gen_frames.py**. Make sure to mention the video path in the code this time.

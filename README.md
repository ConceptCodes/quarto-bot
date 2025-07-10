# Quarto Bot

<p align="center">
  <img src="assets/preview.png" alt="Quarto Bot Demo" width="400"/>
</p>

Inspired by a post on X from the LeRobot Hackathon, where a team trained the [SO-Arm101](https://github.com/TheRobotStudio/SO-ARM100?tab=readme-ov-file) to pick up and place chess pieces, I decided to take on a similar challenge. Rather than simply replicating their work, I wanted to push the concept further: my goal was to train the SO-Arm101 not only to pick-n-place game pieces, but also to understand and play the game itself. Specifically, I set out to teach the SO-Arm101 to play [Quarto](https://en.wikipedia.org/wiki/Quarto_(board_game)), transforming it from a simple robotic arm into an interactive game-playing bot.

## Project Overview
The project involves several key components:
- **Robot Control**: Training the SO-Arm101 to pick up and place pieces on the board.
- **Computer Vision**: Enabling the bot to recognize pieces and their positions on the board
- **ML Strategy**: Developing a Reinforcement Learning (RL) agent that can make strategic decisions based on the current game state.

## Robot Control 

This section is a work in progress.

---
## Computer Vision

### Object Detection
To identify the game pieces, I decided to fine-tune a YOLOv11 model using the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics?tab=readme-ov-file) library. The training process involved the following steps:

1. **Collect and Annotate Images**: I captured a diverse set of images of the Quarto pieces in various positions and lighting conditions. I then annotated each image with bounding boxes for every piece using [Roboflow](https://roboflow.com/). But you can also use [LabelImg](https://github.com/tzutalin/labelImg).

2. **Prepare Dataset and Configuration**: I exported the annotated dataset in YOLO format and created a YAML configuration file specifying the dataset paths, class names (labels), and model parameters.

3. **Train the Model**: I used the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics?tab=readme-ov-file) library to train the model, adjusting hyperparameters such as image size, number of epochs, and batch size to optimize performance.

4. **Evaluate and Iterate**: After training, I evaluated the model’s accuracy on a validation set and refined the dataset or tweaked parameters as needed to improve detection results. (Still working on this part)

#### Training Log
- 07-10-2025
  - Started training the YOLOv11 model with 100 epochs and a batch size of 8.
  - Initial learning rate set to 0.001, patience for early stopping set to 20 epochs.
    - This actually kicked in after 30 epochs, indicating the model was not improving significantly.
  - Overall Performance
    - mAP50: 0.726 (72.6%) - Good overall detection at 50% IoU threshold
    - mAP50-95: 0.664 (66.4%) - Solid performance across stricter IoU thresholds
    - Overall Precision: 0.567 (56.7%) - Moderate precision (some false positives)
    - Overall Recall: 0.847 (84.7%) - Good recall (finding most pieces)
  - Noticed many issues, especially with following pieces:
    - `short-hollow-light-circle`  mAP50=0.35 (very low)
    - `tall-hollow-dark-circle`: Precision=0.318 (many false positives)
    - `tall-hollow-dark-square`: mAP50=0.497 (below average)
    - `tall-solid-dark-circle`: mAP50=0.398 (poor detection)
  - The inference speed was 3267.5 ms, which is quite slow for real-time detection.
  - I will need to gather more data, especially for the poorly performing pieces, and consider adjusting the model architecture or hyperparameters to improve accuracy.

![preview results](assets/07-10-25/results.png)
| Before                       | After                                          |
|------------------------------|------------------------------------------------|
| ![Before](assets/raw.png)    | ![After](assets/07-10-25/detection_result.png) |

### Board Position Detection


---
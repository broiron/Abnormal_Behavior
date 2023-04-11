# Abnormal Behavior Detection using FastMOT
In this project, by using tracker (FastMOT), We tried to detect the customers abnormal behavior in market. <br>
We use two USB Camera and two trackers for each camera are connected with each other. So that we can track customers on different videoes.

## Inspired by

- https://github.com/GeekAlexis/FastMOT
- http://tacodataset.org/

## Requirements

- You have to install the same requirements with FastMOT thing. Or you can use `requirement.txt`
- Please refer to https://github.com/GeekAlexis/FastMOT/blob/master/README.md and get pre-trained network weight.
- Need taco dataset pretrained yolo weight.

## Program Feature
1. Tracking same customer in different cameras.
2. Detect Cluster customers in market.
3. Detect Staying customers in market.
4. Detect the customer who throw away waste on market.

## Testing Video Link
- Customer enter: https://youtu.be/ltUyjvQiipo
- Detect customer cluster: https://youtu.be/KnprahpWc0k
- Detect littering: https://youtu.be/nkj5685UTd4
- Detect customer staying: https://youtu.be/PPEtUZGK1yM
- Detect customer falldown: https://youtu.be/ej0EeAZ0lzs
- Detect aggressive customer: https://youtu.be/Coom_mWVoZU

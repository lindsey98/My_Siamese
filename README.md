# My_Siamese

## Instructions
- Step 1: Clone this repository by ```git clone https://github.com/lindsey98/My_Siamese.git```
- Step 2: Install requirements by 
```pip install -r requirements.txt```
- Step 3: Download targetlist_fit folder from: https://drive.google.com/drive/folders/1rGylKlB4r-U3c8ptdR-q4gsDkpXGNdR8?usp=sharing. Move it under data directory. 
- Step 4: If you want to make prediction for a single logo
```cd scripts```
```python predict.py -p ../data/targetlist_fit/Adobe/0.png -m ../model/resnetv2_rgb.pth
```
- Step 5: If you want to get predictions for all logos, please run 
```cd scripts```
```python dataloader.py -t ../data/targetlist_fit -tl ../data/targetlist_labeldict.pkl -m ../model/resnetv2_rgb.pth
```
## Project structure
- model: 
  - resnetv2_rgb.pth
- scripts: code
- data: 
  - targetlist_fit
  - targetlist_labeldict.pkl
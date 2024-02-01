# Give Me Chocolate

Chocolate detector. Runs on CPU, maybe on GPU if you have it. 
Takes images of chocolate and segments seperate pieces and returns the centers for a robot to pick it and then to eat it.

To get chocolate first do this:

1.
    ```bash
    conda create -n Choko_FastSAM python=3.9
    ```

2. 
    ```bash
    conda activate Choko_FastSAM
    ```

3. 
    ```bash
    cd Give-Me-Chocolate
    pip install -r requirements.txt
    ```
    (To obtain chocolate, you need pytorch>=1.7 and torchvision>=0.8.)

4. 
    ```bash
    pip install git+https://github.com/openai/CLIP.git
    ```

 Then to get chocolate centers run:
  ```bash
    python GetBestChoco.py
  ```

If you want to change the area range of chocolate, open GetChocoArea.py and put desired image in IMAGE_PATH. Then run:
 ```bash
    python GetChocoArea.py
  ```
It will print area of each displayed chocolate.










************************ Dodatno samo za doloceni Raspberry Pi 5 ************************
```bash
cd ~/Projects/GiveMeChocolate/Give-Me-Chocolate
```
```bash
source venv/bin/activate
```
```bash
python RunFastSAM.py
```


Adapted from: [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)

The End

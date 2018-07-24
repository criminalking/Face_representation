# Face representation and similarity

This repo is used to extract feature of face image and compute cosine similarity and l2 distance of two face images.
We tried three state-of-the-art methods to extract feature:

- [Do We Really Need to Collect Million of Faces for Effective Face Recognition?](https://talhassner.github.io/home/publication/2016_ECCV_1). We use the output of pool5 as our feature.
- [LightCNN](https://github.com/AlfredXiangWu/LightCNN)
- [Facenet](https://github.com/davidsandberg/facenet)

# Dependencies

Please install [caffe](http://caffe.berkeleyvision.org/) for Do we, and then run:

```bash
pip install -r requirements.txt
```

for other python packages.

Models used for test please download in [Facenet](https://github.com/davidsandberg/facenet) and [ResNet-101 for Face Recognition](https://docs.google.com/forms/d/e/1FAIpQLSdterS7LCr2hVb-MJWhbdI6AgDDvN0qL45CptoGCbFMbt1F8g/viewform). You do not need to put LightCNN model in model/.


# Run

- Try LightCNN:

```bash
python face_represent_lightcnn.py --img_list=input/list.txt --model="LightCNN-29v2" --num_classes=80013
```

Model here could also be LightCNN-29 or LightCNN-9.

- Try Facenet:

```bash
python face_represent_facenet.py model/20180402-114759.pb
```

- Try Do we ...:

```bash
python face_represent_dowe.py
```


# Results
Facenet is the best!
Speed: LightCNN > Dowe > Facenet


# TODO
- [ ] Crop image with face detector
- [ ] Show results

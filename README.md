# hk_license_plate_replacer
Extracts existing license plate and replaces it in car image with randomly selected HK license plate

### To run

```sh
python2 plate_stitch.py
```

### Run with arguments to specify specific options
#### '--lp_dir_single' : Input Single Hong Kong License Plate image dir
#### '--lp_dir_double' : Input Double Hong Kong License Plate image dir
#### '--img_dir' : Input test image dir
#### '--label_json' : Plate labeling results in json
#### '--plate_dir' : Output extracted plate image dir
#### '--stitched_dir' : Output stitched image dir
#### '--label_txt' : Output OCR label txt
#### '--stitched_label_txt' : Output Stitched OCR label txt
#### '--stitched_json' : Output Stitched json file
```sh
python2 plate_stitch.py --lp_dir_single `pwd`/hklp_single/ --lp_dir_double `pwd`/hklp_double/ --img_dir `pwd`/car_crop_20190505/ --label_json `pwd`/20190505_HK_Double_Plates.json --plate_dir `pwd`/plates/ --stitched_dir `pwd`/stitched/ --label_txt `pwd`/label.txt --stitched_label_txt `pwd`/stitched_label.txt --stitched_json `pwd`/stitched.json
```

#### Repo comes with example synthetically generated single and double row HK license plates
#### Repo comes with example car images with a label.txt file that contains license plate info and a json file containing 4 corner points of license plates for each image
#### Due to the number of potential arguements, it may be easier for you if you change the arguments default in plate_stitch.py in the function parse_args()

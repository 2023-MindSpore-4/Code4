# Scene-Text Oriented Referring Expression Comprehension
This is the official implementation of the Scene Text Awareness Network (STAN) using mindspore(2.0)

## Data Preparation

1. Download the images from [Google Drive](https://drive.google.com/drive/folders/1doQ__aVFvQDqE84AktIf7uc7WxkTOw8B?usp=share_link) and place them in  `ln_data/other/images/reftext`
2. Download the Google OCR results from [Google Drive](https://drive.google.com/drive/folders/1doQ__aVFvQDqE84AktIf7uc7WxkTOw8B?usp=share_link) and place them in `ln_data/ocr`

The folder structure for the dataset is shown below.

```
STAN
└── ln_data
    ├── ocr
    │   └── google_ocr_results_reftext_rank_aggr.json
    └── other
        └── images
            └── reftext (contains 4,594 images)
```




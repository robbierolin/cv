# Style-transfer

## Examples
![Wave](examples/wave.gif)

![Udnie sunset](examples/udnie_senset.jpg)

![](examples/sunset2.gif)

## Usage

For images:
```
python style_transfer.py --image PATH/TO/IMAGE --style STYLE --outdir PATH/TO/OUTPUT/FOLDER
```

For videos:
```
python style_tranfer_video.py --video PATH/TO/VIDEO --style STYLE --outdir PATH/TO/OUTPUT/FOLDER
```

where `STYLE` can have one of the following values:
- [udnie](https://upload.wikimedia.org/wikipedia/en/8/82/Francis_Picabia%2C_1913%2C_Udnie_%28Young_American_Girl%2C_The_Dance%29%2C_oil_on_canvas%2C_290_x_300_cm%2C_Mus%C3%A9e_National_d%E2%80%99Art_Moderne%2C_Centre_Georges_Pompidou%2C_Paris..jpg)
- [wreck](https://images.fineartamerica.com/images-medium-large-5/the-wreck-of-the-amsterdam-flemish-school.jpg)
- [wave](https://upload.wikimedia.org/wikipedia/commons/a/a5/Tsunami_by_hokusai_19th_century.jpg)
- [scream](https://www.1st-art-gallery.com/frame-preview/16889591.jpg?sku=Unframed&thumb=0&huge=0)
- [rain-princess](https://afremov.com/images/product/RAIN-PRINCESS.jpg)
- [la-muse](https://imgc.artprintimages.com/img/print/la-muse_u-l-ejv1k0.jpg?h=550&w=550)

from imgaug import augmenters as iaa


def _augment_geometric(
    d_scale=0.05,
    d_translate=0.02,
    d_rotate=3,
    d_shear_x=5,
    d_shear_y=1,
):
    return iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Affine(
            scale=(1 / (1 + d_scale), (1 + d_scale)),
            translate_percent=dict(
                x=(-d_translate, d_translate),
                y=(-d_translate, d_translate),
            ),
            rotate=(-d_rotate, d_rotate),
        )),
        iaa.Sometimes(0.5, iaa.ShearX(shear=(-d_shear_x, d_shear_x))),
        iaa.Sometimes(0.5, iaa.ShearY(shear=(-d_shear_y, d_shear_y))),
    ], random_order=True)


def augmenter():
    return iaa.Sequential([
        _augment_geometric(),
    ])

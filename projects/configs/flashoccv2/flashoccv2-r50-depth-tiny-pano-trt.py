_base_ = ['./flashoccv2-r50-depth-tiny-pano.py',
          ]

model = dict(
    wocc=True,
    wdet3d=False,
)

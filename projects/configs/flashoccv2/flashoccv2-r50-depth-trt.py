_base_ = ['./flashoccv2-r50-depth.py',
          ]

model = dict(
    wocc=True,
    wdet3d=False,
)

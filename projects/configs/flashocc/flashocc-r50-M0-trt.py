_base_ = ['./flashocc-r50-M0.py',
          ]

model = dict(
    wocc=True,
    wdet3d=False,
)

_base_ = ['./flashocc-r50.py',
          ]

model = dict(
    wocc=True,
    wdet3d=False,
)

from detecto import core, utils, visualize

model = core.Model.load('avocado_weights.pth', ['fit_avocado', 'unfit_avocado'])
# Specify the path to your image
image = utils.read_image('images/13.jpg')
predictions = model.predict(image)

# predictions format: (labels, boxes, scores)
labels, boxes, scores = predictions

# ['alien', 'bat', 'bat']
print(labels) 

#           xmin       ymin       xmax       ymax
# tensor([[ 569.2125,  203.6702, 1003.4383,  658.1044],
#         [ 276.2478,  144.0074,  579.6044,  508.7444],
#         [ 277.2929,  162.6719,  627.9399,  511.9841]])
print(boxes)

# tensor([0.9952, 0.9837, 0.5153])
print(scores)
visualize.show_labeled_image(image, boxes, labels)
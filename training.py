from detecto import core, utils, visualize

dataset = core.Dataset('images/')
model = core.Model(['fit_avocado', 'unfit_avocado'])
model.fit(dataset)

image = utils.read_image('images/10.jpg')
predictions = model.predict(image)

# predictions format: (labels, boxes, scores)
labels, boxes, scores = predictions
model.save('avocado_weights.pth')

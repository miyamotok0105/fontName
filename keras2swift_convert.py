# path = './model.h5'

# import coremltools
# coreml_model = coremltools.converters.keras.convert(path,
#         input_names = 'image',
#         image_input_names = 'image',
#         class_labels = 'labels.txt')

# coreml_model.save('Momomind.mlmodel')

import coremltools
coreml_model = coremltools.converters.keras.convert(model)
coreml_model.save("keras_model.mlmodel")

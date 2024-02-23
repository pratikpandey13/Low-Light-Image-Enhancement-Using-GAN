# Generator model
# Basically a customized U-Net Architecture


def downsample(filters, size, apply_batchnorm=True):   	# Downsampling operation in encoder consist of CONV -> BatchNorm -> LeakyReLU
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  #Convolution Part 
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
  
  #Batch Normalization Part
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  
  #Leaky Relu Filter
  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):		# Upsampling operation in decoder consist of TransposeConv -> BatchNorm -> Dropout
  #Initializer
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()

  #Transpose Convolutional Part
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  #Adding Batch Normalization
  result.add(tf.keras.layers.BatchNormalization())
 
  #applying DropOut
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  #Adding ReLu Filter
  result.add(tf.keras.layers.ReLU())

  return result




# Generator Model is Basically A Sequence of Downsamplings followed by Upsamplings 

def Generator():										# Generator model
  inputs = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = [										# Encoder
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [											# Decoder
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  #initializer
  initializer = tf.random_normal_initializer(0., 0.02)

  #Last Layer of the model 
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []											# Skip connections between encoder and decoder
  for down in down_stack:
    x = down(x)
    skips.append(x)

  # What this part does ?
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])		# Concatenation using skip connections

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
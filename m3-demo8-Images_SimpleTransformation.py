from __future__ import absolute_import, division, print_function

import tensorflow as tf

original_image_list = ["./images/1.jpg",
                       "./images/2.jpg",
                       "./images/3.jpg",
                       "./images/4.jpg",
                       "./images/5.jpg",
                       "./images/6.jpg",
                       "./images/7.jpg"]

# Make a queue of file names including all the images specified.
filename_queue = tf.train.string_input_producer(original_image_list)

# Read an entire image file.
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_list = [];
    for i in range(len(original_image_list)):
        # Read a whole file from the queue, the first returned value in the tuple is the
        # filename which we are ignoring.
        _, image_file = image_reader.read(filename_queue)

        # Decode the image as a JPEG file, this will turn it into a Tensor which we can
        # then use in training.
        image = tf.image.decode_jpeg(image_file)

        # Get a tensor of resized images.
        image = tf.image.resize_images(image, [224, 224])
        image.set_shape((224, 224, 3))

        # Get an image tensor and print its value.
        image_array = sess.run(image)

        # Converts a numpy array of the kind (224, 224, 3) to a Tensor of shape (224, 224, 3)
        image_tensor = tf.stack(image_array)

        image_list.append(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

    # Converts all tensors to a single tensor with a 4th dimension
    # 4 images of (224, 224, 3) can be accessed as (0, 224, 224, 3),
    # (1, 224, 224, 3), (2, 224, 224, 3) etc.
    images_tensor = tf.stack(image_list)

    writer = tf.summary.FileWriter('./m3_demo8', graph=sess.graph)

    # Write out all the images in one go
    summary_str = sess.run(tf.summary.image("images", images_tensor))
    summary_str = sess.run(tf.summary.image("images", images_tensor, max_outputs=10))
    writer.add_summary(summary_str)

    writer.close()

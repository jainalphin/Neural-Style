from tensorflow.compat import v1 as tfv1
from python_files.style import *
from python_files.nst_utils import *
import cv2
def pred(CONTENT_FILENAME,STYLE_FILENAME):
    tf.compat.v1.disable_eager_execution()
    tfv1.reset_default_graph()
    sess = tfv1.InteractiveSession()

    reduce_dims(CONTENT_FILENAME)
    content_image = cv2.imread(CONTENT_FILENAME)
    content_image = reshape_and_normalize_image(content_image)

    reduce_dims(STYLE_FILENAME)
    style_image = cv2.imread(STYLE_FILENAME)
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)
    print("Images loaded...")

    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    model.save('model.h5')
    print("Model loaded...")

    sess.run(model['input'].assign(content_image))

    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out
    J_content = compute_content_cost(a_C, a_G)
    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(sess, model, STYLE_LAYERS)

    J = total_cost(J_content, J_style, alpha=10, beta=40)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.3)
    train_step = optimizer.minimize(J)
    ("training..")

    sess.run(tfv1.global_variables_initializer())
    sess.run(model['input'].assign(generated_image))
    for i in range(22):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
    save_image('static/images/output/generated_image.jpg', generated_image)
    saved_image_path = 'static/images/output/generated_image.jpg'
    return saved_image_path
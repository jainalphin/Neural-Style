import tensorflow as tf
from PIL import Image
def compute_content_cost(a_C,a_G):
  m,n_H,n_W,n_C = tf.convert_to_tensor(a_G).get_shape().as_list()
  a_C_unrolled = tf.reshape(a_C,shape = [m,n_H*n_W,n_C])
  a_G_unrolled = tf.reshape(a_G,shape = [m,n_H*n_W,n_C])

  J_content = 1/(4 * n_H * n_W * n_C) * tf.reduce_sum((a_C-a_G)**2)
  return J_content

def gram_matrix(A):
    GA = tf.matmul(A,tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = tf.convert_to_tensor(a_S).get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, shape=[n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[n_H * n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = 1 / (4 * n_C ** 2 * (n_H * n_W) ** 2) * tf.reduce_sum((GS - GG) ** 2)
    return J_style_layer


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(sess,model,STYLE_LAYERS):
  J_style = 0
  for layer_name, coeff in STYLE_LAYERS:
    out = model[layer_name]
    a_S = sess.run(out)
    a_G = out
    J_style_layer = compute_layer_style_cost(a_S, a_G)
    J_style += coeff * J_style_layer
  return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    return J

def model_nn(sess, model, input_image, num_iterations = 140):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            #save_image("output/" + str(i) + ".png", generated_image)
    save_image('output/generated_image.jpg', generated_image)

    return generated_image

def reduce_dims(image_path):
    imageFile = image_path
    im1 = Image.open(imageFile)
    # adjust width and height to your needs
    width = 400
    height = 300
    im5 = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
    ext = ".jpg"
    im5.save(image_path)


from flask import Flask, render_template,request,redirect
import tensorflow
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

model1 = tf.keras.models.load_model("models/Noleaf")
leaf_name = ['Leaf', 'Noleaf']

model = tf.keras.models.load_model("models/2")

class_names = ['Suffering from Bacterial_spot : Preventative sprays of copper and cultural controls may be used to manage the spread of the disease.Don’t irrigate with sprinklers, since the bacteria can be splashed onto other plants. And be sure to select seeds and transplants that are certified to be free of the disease if this has been a problem for you in the past.',
 'Suffering from Early blight : Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable.',
 'Suffering from  Late blight : You should ruthlessly cull any infected plants and remove them from your property. Just to be safe, you should also remove any plants nearby that could be infected, even if they are not showing symptoms.',
 'Suffering from Leaf Mold : Apply a fungicide according to the manufacturers instructions at the first sign of infection.',
 'Suffering from Septoria leaf spot : An organic fungicide which works against septoria leaf spot is copper fungicide. There is many copper-based fungicidal sprays on the market. Ideally, a copper diammonia diacetate complex is best for treatment. Copper octanoate/copper soap may also work but is a weaker treatment method.',
 'Suffering from Spider mites Two_spotted_spider_mite   : The best way to begin treating for two-spotted mites is to apply a pesticide specific to mites called a miticide. Start treating for two-spotted mites before your plants are seriously damaged. Apply the miticide for control of two-spotted mites every 7 days or so.',
 'Suffering from Target Spot : Many fungicides are registered to control of target spot on tomatoes. Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials. Consult regional disease management guides for recommended products.',
 'Suffering from Yellow Leaf Curl Virus :  Imidacloprid should be sprayed on the entire plant and below the leaves; eggs and flies are often found below the leaves. Spray every 14-21 days and rotate on a monthly basis with Abamectin so that the whiteflies do not build-up resistance to chemicals.',
 'Suffering from Mosaic virus. Because tomatoes are in the same plant family as tobacco nightshades tobacco users can transmit a mosaic virus to their tomato plants simply by touching them While mosaic viruses will nott kill your plant they will weaken them and reduce your crop which is almost as bad You can spot a mosaic virus by the mottled coloring on the leaves or fruit with raised almost blister like spots Do nott allow smoking near your garden and wash your hands or glove them before tending tomatoes if you are a smoker',
 'healthy']

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    return render_template("index.html")

@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        imagefile = request.files["imagefile"]

        if imagefile.filename == "":
            print("File name is invalid")
            return redirect(request.url)
        imagefile = request.files["imagefile"]
        image_path = "./static/" + imagefile.filename
        imagefile.save(image_path)
        image = load_img(image_path, target_size = (256,256))
        image = img_to_array(image)
        image = np.expand_dims(image,0)
        pred  = model1.predict(image)
        pred_class = leaf_name[np.argmax(pred[0])]
        if pred_class == "Noleaf":
            prediction = "Your image was not understood,Please take a close and a clear image of a tomato leaf , showing its detailed features."
            return render_template( "index.html", image_path = image_path, ans =  prediction )
        prediction1 = model.predict(image)
        pred_d = class_names[np.argmax(prediction1[0])]

        confidence = round(100 * (np.max(prediction1[0])), 2)
        
      
    return render_template("index.html", prediction = pred_d, image_path = image_path, Confidence =confidence)

if __name__ == "__main__": 
    app.run(debug = True)
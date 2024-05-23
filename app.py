from fileinput import filename
from flask import *
from model import find_deficiency,fertilizer,get_nutrient_actions

app = Flask(__name__)

@app.route('/')
def main():
	return render_template("index.html")

@app.route('/', methods = ['POST'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		f.save(f.filename)
		pred,max_deficit=find_deficiency(f.filename)
		recommendation=fertilizer(pred)[0]
		print(max_deficit.split(" ")[0])
		actions=get_nutrient_actions(max_deficit.split(" ")[0])
		action1=actions[0]
		action2=actions[1]
		action3=actions[2]
		return render_template("ack.html",boron=pred[0],calcium=pred[1],
		                       healthy=pred[2],iron=pred[3],magnesium=pred[4],manganese=pred[5],
													            potassium=pred[6],sulphur=pred[7],zinc=pred[8],max_deficient=max_deficit,recommendation=recommendation,
																action1=action1,action2=action2,action3=action3)

if __name__ == '__main__':
	app.run()

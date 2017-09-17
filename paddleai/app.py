'''from flask import Flask, request
from twilio import twiml

app = Flask(__name__)

@app.route('/sms', methods=['POST'])
def sms():
	number = request.form['From']
	message_body = request.form['Body']

	resp = twiml.Response()
	resp.message('Hello {}, you said: {}'.format(number, message_body))
	return str(resp)
	

if __name__ == '__main__':
	app.run(debug=True)'''

from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
	message_body = request.form['Body']

	# PADDLEPADDLE

	print(message_body)
		
	f_out = open('/home/noor/Downloads/paddle_ai/out.txt', 'a')
	f_out.write('\n' + message_body)
	f_out.close()

	return ''

if __name__ == "__main__":
	app.run(debug=True)

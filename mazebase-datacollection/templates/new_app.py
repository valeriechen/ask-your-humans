from flask import Flask, render_template, request, jsonify, make_response, json
from pusher import pusher

app = Flask(__name__)

pusher = pusher_client = pusher.Pusher(
  app_id='PUSHER_APP_ID',
  key='PUSHER_APP_KEY',
  secret='PUSHER_APP_SECRET',
  cluster='PUSHER_APP_CLUSTER',
  ssl=True
)

name = ''

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/play')
def play():
  global name
  name = request.args.get('username')
  return render_template('play.html')

@app.route("/pusher/auth", methods=['POST'])
def pusher_authentication():
  auth = pusher.authenticate(
    channel=request.form['channel_name'],
    socket_id=request.form['socket_id'],
    custom_data={
      u'user_id': name,
      u'user_info': {
        u'role': u'player'
      }
    }
  )
  return json.dumps(auth)

# called from guide 
@app.route("/sendMessage", methods=['POST'])
def send_message():
  user_id = request.form['user_id']
  game_id = request.form['game_id']
  message = request.form['message']

  #email_yourself
  save(user_id, game_id, message, game_state)

  map.put(game_id, message, False)#is_read is false


  return json.dumps(auth)

#called from player
@app.route("/readMessage", methods=['POST'])
def send_message():
  user_id = request.form['user_id']
  game_id = request.form['game_id']

  if map.get(get_id).not_empty():
    message = whatever you get :) 

  return json.dumps({'message': message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

name = ''

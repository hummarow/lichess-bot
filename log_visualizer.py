
from flask import Flask, render_template, jsonify, request
import os
import json
import chess
import chess.svg

app = Flask(__name__)

LOG_DIR = 'game_logs'

@app.route('/')
def index():
    games = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.json')], reverse=True)
    return render_template('index.html', games=games)

@app.route('/game/<game_id>')
def game(game_id):
    log_path = os.path.join(LOG_DIR, game_id)
    with open(log_path, 'r') as f:
        game_data = json.load(f)
    return render_template('game.html', game_id=game_id, game_data=game_data)

@app.route('/get_move_data', methods=['POST'])
def get_move_data():
    game_id = request.json['game_id']
    move_index = request.json['move_index']
    
    log_path = os.path.join(LOG_DIR, game_id)
    with open(log_path, 'r') as f:
        game_data = json.load(f)
        
    move_data = game_data['moves'][move_index]
    
    fen_before_move = move_data['fen_before_move']
    fen_after_move = move_data['fen_after_move']
    move_uci = move_data['move_uci']
    player = move_data['player']
    llm_response = move_data['llm_response']

    board_before_svg = chess.svg.board(board=chess.Board(fen_before_move), size=400)
    board_after_svg = chess.svg.board(board=chess.Board(fen_after_move), size=400, lastmove=chess.Move.from_uci(move_uci))
    
    return jsonify({
        'board_before_svg': board_before_svg,
        'board_after_svg': board_after_svg,
        'move_uci': move_uci,
        'player': player,
        'llm_response': llm_response
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)

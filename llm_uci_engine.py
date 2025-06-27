#!/usr/bin/env python3
import chess
import chess.engine
import random
import logging
import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import openai
from pydantic import BaseModel, Field, ValidationError

# Setup logging for the UCI engine
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='llm_uci_engine.log',
                    filemode='w')
logger = logging.getLogger(__name__)

class GameLogger:
    def __init__(self, log_dir="game_logs"):
        self.log_dir = log_dir
        self.game_id = None
        self.move_entries = [] # List of dictionaries, each representing a single move
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def new_game(self):
        self.game_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.move_entries = []
        self.save_log(chess.Board()) # Pass an empty board for initial save

    def log_move(self, board: chess.Board, move_uci: str, player: str, fen_before_move: str, fen_after_move: str, llm_response: "ChessMoveResponse" = None):
        log_entry = {
            "move_number": len(self.move_entries) // 2 + 1, # Full move number
            "player": player,
            "move_uci": move_uci,
            "fen_before_move": fen_before_move,
            "fen_after_move": fen_after_move,
            "llm_response": llm_response.model_dump() if llm_response else None,
        }
        self.move_entries.append(log_entry)
        self.save_log(board)

    def save_log(self, board: chess.Board):
        if self.game_id:
            log_path = os.path.join(self.log_dir, f"{self.game_id}.json")
            pgn_string = board.variation_san([move for move in board.move_stack])

            with open(log_path, "w") as f:
                json.dump({
                    "pgn": pgn_string,
                    "moves": self.move_entries
                }, f, indent=2)


class ChessMoveResponse(BaseModel):
    move: str = Field(..., description="The optimal chess move in Standard Algebraic Notation (SAN) format (e.g., 'e4', 'Nf3', 'O-O').")
    reason: str = Field(..., description="A brief explanation for the chosen move.")
    principle_variation: str = Field(..., description="The main line of play the LLM considered, in UCI format (e.g., 'e2e4 e7e5 g1f3').")
    tactic: str = Field(..., description="Any immediate tactical ideas or threats related to the chosen move.")
    strategy: str = Field(..., description="The long-term strategic goal or plan behind the chosen move.")

class LLMEngine:
    """
    A chess engine that uses a Large Language Model (LLM) to choose moves.
    Supports OpenAI and Anthropic (Claude) models.
    """
    def __init__(self, llm_type: str, model_name: str):
        logger.debug(f"LLMEngine __init__ called with llm_type={llm_type}, model_name={model_name}")
        load_dotenv()  # Load environment variables from .env file
        self.llm_type = llm_type
        self.model_name = model_name
        self.client = None
        self.message_history = [] # Initialize message history
        self.game_logger = GameLogger()

        try:
            if self.llm_type == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set.")
                self.client = openai.OpenAI(api_key=api_key)
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}. Choose 'openai'.")
            logger.debug("LLM client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise # Re-raise to indicate initialization failure

    def _san_to_uci(self, board: chess.Board, san_move: str) -> str | None:
        try:
            move = board.parse_san(san_move)
            return move.uci()
        except chess.IllegalMoveError:
            logger.warning(f"Illegal SAN move: {san_move} for board {board.fen()}")
            return None
        except chess.InvalidMoveError:
            logger.warning(f"Invalid SAN move format: {san_move}")
            return None

    def _get_llm_response(self, messages: list[dict]) -> ChessMoveResponse:
        logger.debug(f"_get_llm_response called. LLM Type: {self.llm_type}, Model: {self.model_name}")
        try:
            if self.llm_type == "openai":
                raw_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=200, # Increased max_tokens to accommodate new fields
                    temperature=0.5,
                )

                logger.debug(f"OpenAI input messages: {messages}")
                raw_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=100, # Increased max_tokens to accommodate reason
                    temperature=0.5, # Increased temperature for more varied responses
                )
                logger.debug(f"OpenAI raw response object: {raw_response}")
                messages.append({
                    "role": "assistant",
                    "content": raw_response.choices[0].message.content
                })
                # Parse the raw response into the Pydantic model
                parsed_response = self.client.responses.parse(
                    model=self.model_name,
                    input=messages,
                    text_format=ChessMoveResponse,
                )
                logger.debug(f"Parsed response: {parsed_response}")
                return parsed_response.output[0].content[0].parsed
            raise ValueError("Unsupported LLM type for _get_llm_response.")
        except Exception as e:
            logger.error(f"Error during LLM API call: {e}")
            raise # Re-raise to be caught by the calling search method

    def get_board_info(self, board: chess.Board, pgn: str) -> str:
        """
        Returns a string representation of the current board state, including FEN, turn, castling rights,
        en passant square, halfmove clock, fullmove number, and whether the king is in check.
        """
        fen = board.fen()
        turn = "White" if board.turn == chess.WHITE else "Black"
        castling_rights = board.castling_rights
        ep_square = chess.square_name(board.ep_square) if board.ep_square else "None"
        halfmove_clock = board.halfmove_clock
        fullmove_number = board.fullmove_number
        is_check = board.is_check()

        return (f"Game Record (PGN):\n{pgn}\n\n"
                f"Current FEN: {fen}\n"
                f"Turn: {turn}\n"
                f"Castling Rights: {castling_rights}\n"
                f"En Passant Square: {ep_square}\n"
                f"Halfmove Clock: {halfmove_clock}\n"
                f"Fullmove Number: {fullmove_number}\n"
                f"Is King in Check: {is_check}\n")

    def get_best_move(self, board: chess.Board) -> chess.Move:
        logger.debug("get_best_move called.")
        fen = board.fen()
        legal_moves = {move.uci() for move in board.legal_moves}

        system_message_content = (
            "You are a chess grandmaster. Your goal is to play optimal chess moves.\n"
            "Consider king safety, material balance, center control, piece development, and tactical threats.\n"
            "Provide only the optimal chess move, a brief reason, the principle variation, any immediate tactical ideas, and the long-term strategic goal in JSON format."
            "The JSON should conform to the ChessMoveResponse Pydantic model: {{'move': 'e4', 'reason': 'This move controls the center and opens lines for development.', 'principle_variation': 'e2e4 e7e5 g1f3', 'tactic': 'None', 'strategy': 'Control the center and develop pieces.'}}"
        )

        # Initialize message history for a new game or if it's empty
        if not self.message_history or self.message_history[0]["role"] != "system":
            self.message_history = [{"role": "system", "content": system_message_content}]

        retries = 3
        for i in range(retries):
            logger.debug(f"Board's move stack: {board.move_stack}")
            pgn = chess.Board().variation_san([move for move in board.move_stack])
            user_prompt_content = self.get_board_info(board, pgn)
            if i > 0:
                user_prompt_content += f" (Previous attempt returned an invalid move or format. Please provide a valid UCI move from the legal moves: {', '.join(sorted(list(legal_moves)))})\n"
            
            # Append user prompt to history
            self.message_history.append({"role": "user", "content": user_prompt_content})
            one_step_message = [{"role": "system", "content": system_message_content}, {"role": "user", "content": user_prompt_content}]
            try:
                llm_response: ChessMoveResponse = self._get_llm_response(one_step_message)
                # llm_response: ChessMoveResponse = self._get_llm_response(self.message_history)
                llm_move_san = llm_response.move
                llm_move_uci = self._san_to_uci(board, llm_move_san)
                llm_reason = llm_response.reason
                llm_pv = llm_response.principle_variation
                llm_tactic = llm_response.tactic
                llm_strategy = llm_response.strategy
                
                logger.info(f"LLM suggested move: {llm_move_san} (UCI: {llm_move_uci}), Reason: {llm_reason}, PV: {llm_pv}, Tactic: {llm_tactic}, Strategy: {llm_strategy}")

                if llm_move_uci and llm_move_uci in legal_moves:
                    # Append assistant's response to history
                    self.message_history.append({"role": "assistant", "content": json.dumps(llm_response.model_dump())})
                    
                    # Log the LLM's move
                    fen_before_llm_move = board.fen()
                    board.push_uci(llm_move_uci)
                    fen_after_llm_move = board.fen()
                    self.game_logger.log_move(board=board, move_uci=llm_move_uci, player="llm", fen_before_move=fen_before_llm_move, fen_after_move=fen_after_llm_move, llm_response=llm_response)
                    return chess.Move.from_uci(llm_move_uci)
                else:
                    logger.warning(f"LLM returned an illegal move: {llm_move_uci}. Legal moves are: {', '.join(sorted(list(legal_moves)))}")
                    # If illegal move, remove the last user prompt and try again
                    self.message_history.pop() 
            except Exception as e:
                logger.error(f"Error getting LLM response or parsing: {e}")
                # If error, remove the last user prompt and try again
                self.message_history.pop()

        logger.error("LLM failed to provide a valid move after multiple retries. Falling back to a random legal move.")
        return random.choice(list(board.legal_moves))

def main():
    engine = None
    board = chess.Board()
    llm_type_option = "openai"  # Default LLM type
    model_name_option = "gpt-4o"  # Default model name

    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:  # EOF
                logger.debug("EOF received, exiting.")
                break
            logger.debug(f"Received: {line}")

            if line == "uci":
                print("id name LLM Chess Engine")
                print("id author Gemini")
                print("option name LLM_Type type string default openai")
                print("option name Model_Name type string default gpt-4o")
                print("uciok")
                sys.stdout.flush()
                logger.debug("Sent uciok.")
            elif line == "isready":
                if engine is None:
                    logger.debug("Initializing LLMEngine in isready.")
                    try:
                        engine = LLMEngine(llm_type_option, model_name_option)
                    except ValueError as e:
                        logger.error(f"Error initializing LLMEngine: {e}")
                        sys.exit(1) # Exit if LLM engine cannot be initialized
                print("readyok")
                sys.stdout.flush()
                logger.debug("Sent readyok.")
            elif line == "ucinewgame":
                board = chess.Board()
                if engine: # Reset message history for a new game
                    engine.message_history = []
                    engine.game_logger.new_game()
                logger.debug("New game started.")
            elif line.startswith("position"):
                parts = line.split(" ")
                if "startpos" in parts:
                    board = chess.Board()
                    moves_index = parts.index("moves") + 1 if "moves" in parts else len(parts)
                    for move_uci in parts[moves_index:]:
                        fen_before_user_move = board.fen()
                        try:
                            board.push_uci(move_uci)
                            engine.game_logger.log_move(board=board, move_uci=move_uci, player="user", fen_before_move=fen_before_user_move, fen_after_move=board.fen())
                        except ValueError:
                            logger.error(f"Invalid move in position command: {move_uci}")
                    logger.debug(f"Position set from startpos. Current FEN: {board.fen()}")
                elif "fen" in parts:
                    fen_index = parts.index("fen") + 1
                    fen_string = " ".join(parts[fen_index:fen_index+6])
                    board = chess.Board(fen_string)
                    moves_index = parts.index("moves") + 1 if "moves" in parts else len(parts)
                    for move_uci in parts[moves_index:]:
                        fen_before_user_move = board.fen()
                        try:
                            board.push_uci(move_uci)
                            engine.game_logger.log_move(board=board, move_uci=move_uci, player="user", fen_before_move=fen_before_user_move, fen_after_move=board.fen())
                        except ValueError:
                            logger.error(f"Invalid move in position command: {move_uci}")
                    logger.debug(f"Position set from FEN. Current FEN: {board.fen()}")
            elif line.startswith("setoption name"):
                parts = line.split(" ")
                option_name = parts[2]
                option_value = parts[4]
                if option_name == "LLM_Type":
                    llm_type_option = option_value
                    logger.info(f"Set LLM_Type to {llm_type_option}")
                elif option_name == "Model_Name":
                    model_name_option = option_value
                    logger.info(f"Set Model_Name to {model_name_option}")
                # Re-initialize engine if options change and engine was already initialized
                if engine is not None and (engine.llm_type != llm_type_option or engine.model_name != model_name_option):
                    logger.debug("Re-initializing LLMEngine due to option change.")
                    try:
                        engine = LLMEngine(llm_type_option, model_name_option)
                        logger.info("LLMEngine re-initialized with new options.")
                    except ValueError as e:
                        logger.error(f"Error re-initializing LLMEngine: {e}")
                        engine = None # Reset engine if re-initialization fails
            elif line.startswith("go"):
                logger.debug("Received go command.")
                if engine:
                    best_move = engine.get_best_move(board)
                    print(f"bestmove {best_move.uci()}")
                    sys.stdout.flush()
                    logger.debug(f"Sent bestmove {best_move.uci()}")
                else:
                    logger.error("Engine not initialized. Making a random move.")
                    print(f"bestmove {random.choice(list(board.legal_moves)).uci()}")
                    sys.stdout.flush()
                    logger.debug("Sent random bestmove.")
            elif line == "quit":
                logger.debug("Received quit command, exiting.")
                break
        except Exception as e:
            logger.critical(f"Unhandled error in main loop: {e}", exc_info=True)
            break

if __name__ == "__main__":
    main()

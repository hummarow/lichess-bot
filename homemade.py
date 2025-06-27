
# Some example classes for people who want to create a homemade bot.

# With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.

import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import logging
import os
from dotenv import load_dotenv
import openai
import anthropic

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""


# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)


class LLMEngine(ExampleEngine):
    """
    A chess engine that uses a Large Language Model (LLM) to choose moves.
    Supports OpenAI and Anthropic (Claude) models.
    """
    def __init__(self, llm_type: str, model_name: str):
        load_dotenv()  # Load environment variables from .env file
        self.llm_type = llm_type
        self.model_name = model_name
        self.client = None

        if self.llm_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self.client = openai.OpenAI(api_key=api_key)
        elif self.llm_type == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}. Choose 'openai' or 'anthropic'.")

    def _get_llm_response(self, prompt: str, system_message: str) -> str:
        if self.llm_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,  # A move is short (e.g., e2e4)
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        elif self.llm_type == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[
                    {"role": "user", "content": system_message + "\n" + prompt}
                ],
                temperature=0.1,
            )
            return response.content[0].text.strip()
        return ""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        fen = board.fen()
        legal_moves = {move.uci() for move in board.legal_moves}
        system_message = "You are a chess grandmaster. Given a FEN string, provide only the optimal chess move in UCI format (e.g., 'e2e4'). Do not include any other text or explanation."

        retries = 3
        for i in range(retries):
            prompt = f"Current FEN: {fen}"
            if i > 0:
                prompt += f" (Previous attempt returned an invalid move. Please provide a valid UCI move from the legal moves: {', '.join(sorted(list(legal_moves)))})"

            try:
                llm_move_uci = self._get_llm_response(prompt, system_message)
                logger.info(f"LLM suggested move: {llm_move_uci}")

                if llm_move_uci in legal_moves:
                    move = chess.Move.from_uci(llm_move_uci)
                    return PlayResult(move, None)
                else:
                    logger.warning(f"LLM returned an illegal move: {llm_move_uci}. Legal moves are: {', '.join(sorted(list(legal_moves)))}")
            except Exception as e:
                logger.error(f"Error getting LLM response: {e}")

        logger.error("LLM failed to provide a valid move after multiple retries. Falling back to a random legal move.")
        return PlayResult(random.choice(list(board.legal_moves)), None)
